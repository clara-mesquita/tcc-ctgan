"""CLI."""

import argparse

import pandas as pd
from sklearn.metrics import mean_squared_error

import argparse


import numpy as np

import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument('-e', '--epochs', default=300, type=int, help='Number of training epochs')
    parser.add_argument(
        '-t', '--tsv', action='store_true', help='Load data in TSV format instead of CSV'
    )
    parser.add_argument(
        '--no-header',
        dest='header',
        action='store_false',
        help='The CSV file has no header. Discrete columns will be indices.',
    )

    parser.add_argument('-m', '--metadata', help='Path to the metadata')
    parser.add_argument(
        '-d', '--discrete', help='Comma separated list of discrete columns without whitespaces.'
    )
    parser.add_argument(
        '-n',
        '--num-samples',
        type=int,
        help='Number of rows to sample. Defaults to the training data size',
    )

    parser.add_argument(
        '--generator_lr', type=float, default=2e-4, help='Learning rate for the generator.'
    )
    parser.add_argument(
        '--discriminator_lr', type=float, default=2e-4, help='Learning rate for the discriminator.'
    )

    parser.add_argument(
        '--generator_decay', type=float, default=1e-6, help='Weight decay for the generator.'
    )
    parser.add_argument(
        '--discriminator_decay', type=float, default=0, help='Weight decay for the discriminator.'
    )

    parser.add_argument(
        '--embedding_dim', type=int, default=128, help='Dimension of input z to the generator.'
    )
    parser.add_argument(
        '--generator_dim',
        type=str,
        default='256,256',
        help='Dimension of each generator layer. Comma separated integers with no whitespaces.',
    )
    parser.add_argument(
        '--discriminator_dim',
        type=str,
        default='256,256',
        help='Dimension of each discriminator layer. Comma separated integers with no whitespaces.',
    )

    parser.add_argument(
        '--batch_size', type=int, default=500, help='Batch size. Must be an even number.'
    )
    parser.add_argument(
        '--save', default=None, type=str, help='A filename to save the trained synthesizer.'
    )
    parser.add_argument(
        '--load', default=None, type=str, help='A filename to load a trained synthesizer.'
    )

    parser.add_argument(
        '--sample_condition_column', default=None, type=str, help='Select a discrete column name.'
    )
    parser.add_argument(
        '--sample_condition_column_value',
        default=None,
        type=str,
        help='Specify the value of the selected discrete column.',
    )

    parser.add_argument('data', help='Path to training data')
    parser.add_argument('output', help='Path of the output file')

    return parser.parse_args()


# def main():
#     """CLI."""
#     args = _parse_args()
#     if args.tsv:
#         data, discrete_columns = read_tsv(args.data, args.metadata)
#     else:
#         print("About to call read_csv()")
#         data, discrete_columns = read_csv(args.data, args.metadata, args.header, args.discrete)
#         mask = get_null_mask(data)

#     if args.load:
#         model = CTGAN.load(args.load)
#     else:
#         generator_dim = [int(x) for x in args.generator_dim.split(',')]
#         discriminator_dim = [int(x) for x in args.discriminator_dim.split(',')]
#         model = CTGAN(
#             embedding_dim=args.embedding_dim,
#             generator_dim=generator_dim,
#             discriminator_dim=discriminator_dim,
#             generator_lr=args.generator_lr,
#             generator_decay=args.generator_decay,
#             discriminator_lr=args.discriminator_lr,
#             discriminator_decay=args.discriminator_decay,
#             batch_size=args.batch_size,
#             epochs=args.epochs,
#         )

#     col_vazao = data.columns.get_loc("Vazao")
#     col_vazao_bbr = data.columns.get_loc("Vazao_bbr")

#     n_test = data.shape[0]
#     num_missing_vazao = int(n_test * 0.10)
#     num_missing_vazao_bbr = int(n_test * 0.10)

#     missing_rows_vazao = np.random.choice(n_test, num_missing_vazao, replace=False)
#     missing_rows_vazao_bbr = np.random.choice(n_test, num_missing_vazao_bbr, replace=False)

#     incomplete_data = data.copy()
#     incomplete_data.iloc[missing_rows_vazao, col_vazao] = np.nan
#     incomplete_data.iloc[missing_rows_vazao_bbr, col_vazao_bbr] = np.nan

#     model.fit(train_data=incomplete_data, discrete_columns=discrete_columns, epochs=300)

#     if args.save is not None:
#         model.save(args.save)

#     num_samples = args.num_samples or len(incomplete_data)

#     if args.sample_condition_column is not None:
#         assert args.sample_condition_column_value is not None

#      # Preenchendo os valores faltantes no dataset incompleto
#     imputed_data = model.impute(incomplete_data)

#     # Salva os dados imputados no arquivo de saída especificado
#     if args.tsv:
#         # Se for TSV
#         imputed_data.to_csv(args.output, sep='\t', index=False)
#     else:
#         # CSV padrão
#         imputed_data.to_csv(args.output, index=False)

#     print(f"Imputed data saved to {args.output}")

def main():
    args = _parse_args()

    # Carregar os dados originais e imputados
    original_data = pd.read_csv(args.data)
    imputed_data = pd.read_csv(args.output)

    original_data = original_data.drop(columns=['Link_bottleneck'])

    # Identificar índices com NaN no imputed_data
    nan_indices = imputed_data[imputed_data.isna().any(axis=1)].index

    # Remover valores correspondentes em original_data
    original_data = original_data.drop(index=nan_indices)
    imputed_data = imputed_data.dropna()
    original_data = original_data.drop(columns=original_data.columns[0])  
    print("Read data:")
    print(original_data.shape)
    print(imputed_data.shape)
    
    # Verificar se os dados possuem o mesmo formato
    if original_data.shape != imputed_data.shape:
        raise ValueError("Original data and imputed data must have the same shape")

    # Calcular o RMSE por coluna
    rmse_results = {}
    for column in original_data.columns:
        original_values = original_data[column].dropna().values
        imputed_values = imputed_data.loc[original_data[column].notna(), column].values

        rmse = np.sqrt(mean_squared_error(original_values, imputed_values))
        rmse_results[column] = rmse

    # Exibir resultados
    print("RMSE por coluna:")
    for column, rmse in rmse_results.items():
        print(f"{column}: {rmse:.4f}")

    # Salvar resultados em um arquivo
    rmse_df = pd.DataFrame(list(rmse_results.items()), columns=['Column', 'RMSE'])
    rmse_df.to_csv('rmse_results.csv', index=False)
    print("RMSE results saved to 'rmse_results.csv'")


if __name__ == '__main__':
    main()
