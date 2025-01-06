"""CTGAN module."""

import warnings

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Linear, Module, ReLU, functional, LeakyReLU, Dropout, Sequential
from tqdm import tqdm

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state


# class Discriminator(Module):
#     """Discriminator for the CTGAN."""

#     def __init__(self, input_dim, discriminator_dim, pac=10):
#         super(Discriminator, self).__init__()
#         dim = input_dim * pac
#         self.pac = pac
#         self.pacdim = dim
#         seq = []
#         for item in list(discriminator_dim):
#             seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
#             dim = item

#         seq += [Linear(dim, 1)]
#         self.seq = Sequential(*seq)

#     def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
#         """Compute the gradient penalty."""
#         alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
#         alpha = alpha.repeat(1, pac, real_data.size(1))
#         alpha = alpha.view(-1, real_data.size(1))

#         interpolates = alpha * real_data + ((1 - alpha) * fake_data)

#         disc_interpolates = self(interpolates)

#         gradients = torch.autograd.grad(
#             outputs=disc_interpolates,
#             inputs=interpolates,
#             grad_outputs=torch.ones(disc_interpolates.size(), device=device),
#             create_graph=True,
#             retain_graph=True,
#             only_inputs=True,
#         )[0]

#         gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
#         gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

#         return gradient_penalty

#     def forward(self, input_):
#         """Apply the Discriminator to the `input_`."""
#         assert input_.size()[0] % self.pac == 0
#         return self.seq(input_.view(-1, self.pacdim))

class Discriminator(Module):
    def __init__(self, data_dim, cond_dim, discriminator_dim, pac=10, use_mask=False):
        super().__init__()
        self.use_mask = use_mask
        self.pac = pac
        # Se usa data + cond + mask => input_dim = data_dim + cond_dim + data_dim
        # = 2*data_dim + cond_dim
        self.input_dim = (2 * data_dim) + cond_dim if use_mask else (data_dim + cond_dim)
        self.pacdim = self.input_dim * pac

        # Agora criamos camadas baseadas em self.pacdim
        dim = self.pacdim

        seq = []
        for hidden_size in discriminator_dim:
            seq.append(Linear(dim, hidden_size))
            seq.append(LeakyReLU(0.2))
            seq.append(Dropout(0.5))
            dim = hidden_size
        
        seq.append(Linear(dim, 1))
        self.seq = Sequential(*seq)


    def forward(self, data, mask=None, cond=None):
        # data: (B, data_dim)
        # mask: (B, data_dim) ou None
        # cond: (B, cond_dim) ou None
        
        # Se quiser concatenar cond:
        if cond is not None:
            data = torch.cat([data, cond], dim=1)  # (B, data_dim + cond_dim)

        if (mask is not None) and self.use_mask:
            data = torch.cat([data, mask], dim=1)  # + data_dim => shape (B, 2*data_dim + cond_dim)
        
        assert data.size(0) % self.pac == 0

        # (B, self.input_dim) => (B, self.input_dim * pac)
        out = self.seq(data.view(-1, self.pacdim))
        return out

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


# class Generator(Module):
#     """Generator for the CTGAN."""

#     def __init__(self, embedding_dim, generator_dim, data_dim):
#         super(Generator, self).__init__()
#         dim = embedding_dim
#         seq = []
#         for item in list(generator_dim):
#             seq += [Residual(dim, item)]
#             dim += item
#         seq.append(Linear(dim, data_dim))
#         self.seq = Sequential(*seq)

#     def forward(self, input_):
#         """Apply the Generator to the `input_`."""
#         data = self.seq(input_)
#         return data

class Generator(Module):
    def __init__(self, embedding_dim, cond_dim, generator_dim, data_dim):
        super().__init__()
        # Se você concatena x_obs + mask + noise + cond:
        self.input_dim = (data_dim + data_dim) + embedding_dim + cond_dim
        # ou (2 * data_dim) + (embedding_dim + cond_dim)

        dim = self.input_dim
        
        # Monta as camadas
        seq = []
        for hidden_size in generator_dim:
            seq.append(Residual(dim, hidden_size))  # Residual ou Linear
            dim += hidden_size
        
        # Camada final gera data_dim (imputação)
        seq.append(Linear(dim, data_dim))

        self.seq = Sequential(*seq)

    def forward(self, x_obs, mask, noise, cond=None):
        # Se 'cond' não for None, concatene no 'noise'
        # ou concatene tudo de uma vez
        if cond is not None:
            noise = torch.cat([noise, cond], dim=1)  # shape: (B, embedding_dim + cond_dim)

        input_ = torch.cat([x_obs, mask, noise], dim=1)  # (B, 2*data_dim + embedding_dim + cond_dim)
        
        data_generated = self.seq(input_)
        
        # Ex.: se for imputação, faça:
        #   imputed_data = x_obs * mask + data_generated * (1 - mask)
        #   return imputed_data
        return data_generated



def get_incomplete_batch(train_data, batch_size, device='cpu'):
    # 1) Selecione índices aleatórios (ou use DataLoader se preferir)
    idx = np.random.choice(len(train_data), size=batch_size, replace=False)
    
    # 2) Extraia um subset do dataset
    batch = train_data[idx]  # shape: (B, D)
    
    # 3) Crie a mask_batch: 1 se não é NaN, 0 se é NaN
    mask_batch = ~np.isnan(batch)      # boolean array (True/False)
    mask_batch = mask_batch.astype(float)  # converter para 0/1 em float
    
    # 4) Crie x_obs_batch: copie o batch e substitua NaN por 0
    x_obs_batch = np.copy(batch)
    x_obs_batch[np.isnan(x_obs_batch)] = 0.0
    
    # 5) Converter para torch.Tensor
    x_obs_batch = torch.tensor(x_obs_batch, dtype=torch.float32).to(device)
    mask_batch  = torch.tensor(mask_batch,  dtype=torch.float32).to(device)
    
    return x_obs_batch, mask_batch

class CTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        cuda=True,
    ):
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    '`epochs` argument in `fit` method has been deprecated and will be removed '
                    'in a future version. Please pass `epochs` to the constructor instead'
                ),
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions
        cond_dim = self._data_sampler.dim_cond_vec()

        self._generator = Generator(
    embedding_dim=self._embedding_dim, 
    cond_dim=cond_dim,
    generator_dim=self._generator_dim,
    data_dim=data_dim
).to(self._device)

        discriminator = Discriminator(
        data_dim + self._data_sampler.dim_cond_vec(),
        cond_dim=cond_dim,
        discriminator_dim= self._discriminator_dim,
        pac=self.pac,
        use_mask=True    # <--- IMPORTANTE se quiser passar máscara
    ).to(self._device)
        
        print("data_dim =", data_dim)
        print("cond_vec_dim =", self._data_sampler.dim_cond_vec())
        print("discriminator_dim =", self._discriminator_dim)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        print("oi")
        print(discriminator)                    # ver se mostra camadas
        print(list(discriminator.parameters()))

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):

                    x_obs_batch, mask_batch_fake = get_incomplete_batch(
                    train_data,
                    batch_size=self._batch_size,
                    device=self._device
                    )

                    x_obs_real, mask_batch_real = get_incomplete_batch(
                    train_data,
                    batch_size=self._batch_size,
                    device=self._device
                )
                    
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    # fake = self._generator(fakez)
                    # fakeact = self._apply_activate(fake)

                    imputed_data = self._generator(x_obs_batch, mask_batch_fake, fakez)
                    fakeact = self._apply_activate(imputed_data)  # ativação

                    # real = torch.from_numpy(real.astype('float32')).to(self._device)
                    real_data = x_obs_real 
                    
                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real_data, c2], dim=1)
                    else:
                        real_cat = real_data
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat, mask_batch_fake)
                    y_real = discriminator(real_cat, mask_batch_real)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                x_obs_batch, mask_batch_fake = get_incomplete_batch(
                train_data,
                batch_size=self._batch_size,
                device=self._device
            )
                
                fakez = torch.normal(mean=mean, std=std)

                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                imputed_data = self._generator(x_obs_batch, mask_batch_fake, fakez)
                fakeact = self._apply_activate(imputed_data)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1), mask_batch_fake)
                else:
                    y_fake = discriminator(fakeact, mask_batch_fake)


                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(imputed_data, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)


    def impute(self, incomplete_data, batch_size=None, condition_column=None, condition_value=None):
        if batch_size is None:
            batch_size = self._batch_size

        incomplete_data_t = self._transformer.transform(incomplete_data)

        n = len(incomplete_data_t)
        steps = (n // batch_size) + 1
        imputed_result = []

        for step_i in range(steps):
            start = step_i * batch_size
            end = min((step_i + 1) * batch_size, n)
            if start >= end:
                break

            batch_incomplete = incomplete_data_t[start:end]
            mask_batch = ~np.isnan(batch_incomplete)
            x_obs_batch = np.copy(batch_incomplete)
            x_obs_batch[np.isnan(x_obs_batch)] = 0.0

            x_obs_batch_torch = torch.tensor(x_obs_batch, dtype=torch.float32, device=self._device)
            mask_batch_torch = torch.tensor(mask_batch.astype(float), dtype=torch.float32, device=self._device)

            curr_size = x_obs_batch_torch.size(0)
            noise = torch.randn((curr_size, self._embedding_dim), device=self._device)

            if condition_column is not None and condition_value is not None:
                condition_info = self._transformer.convert_column_name_value_to_id(
                    condition_column, condition_value
                )
                global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                    condition_info, self._batch_size
                )
            else:
                global_condition_vec = None

            if condition_column and condition_value:
                cond_vec = global_condition_vec[:curr_size]
                cond_vec = torch.tensor(cond_vec, dtype=torch.float32, device=self._device)
            else:
                cond_vec = None

            if cond_vec is not None:
                imputed_data_torch = self._generator(x_obs_batch_torch, mask_batch_torch, noise, cond_vec)
            else:
                imputed_data_torch = self._generator(x_obs_batch_torch, mask_batch_torch, noise)

            imputed_data_torch = (
                x_obs_batch_torch * mask_batch_torch + imputed_data_torch * (1.0 - mask_batch_torch)
            )
            imputed_data_np = imputed_data_torch.detach().cpu().numpy()
            imputed_result.append(imputed_data_np)

        imputed_result = np.concatenate(imputed_result, axis=0)
        imputed_result = imputed_result[:n]

        # Verificar e ajustar dimensões
        expected_dimensions = self._transformer.output_dimensions
        if imputed_result.shape[1] > expected_dimensions:
            print(f"Trimming imputed_result from {imputed_result.shape[1]} to {expected_dimensions} columns.")
            imputed_result = imputed_result[:, :expected_dimensions]

        print(f"Shape of imputed_result before inverse_transform: {imputed_result.shape}")
        imputed_result = self._transformer.inverse_transform(imputed_result)

        return imputed_result
