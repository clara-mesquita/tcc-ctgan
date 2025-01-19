"""Data loading."""

import json

import os 

import numpy as np
import pandas as pd


def read_csv(csv_filename, meta_filename=None, header=True, discrete=None):
    
    """Read a csv file."""
    data = pd.read_csv(csv_filename, header='infer' if header else None)  
    data = data.drop(columns=data.columns[0])  
    data = data.drop(columns = ['Link_bottleneck'])
    
    if meta_filename:
        with open(meta_filename) as meta_file:
            metadata = json.load(meta_file)

        discrete_columns = [
            column['name'] for column in metadata['columns'] if column['type'] != 'continuous'
        ]

    elif discrete:
        discrete_columns = discrete.split(',')
        if not header:
            discrete_columns = [int(i) for i in discrete_columns]

    else:
        discrete_columns = []

    return data, discrete_columns


def read_tsv(data_filename, meta_filename):
    """Read a tsv file."""
    with open(meta_filename) as f:
        column_info = f.readlines()

    column_info_raw = [x.replace('{', ' ').replace('}', ' ').split() for x in column_info]

    discrete = []
    continuous = []
    column_info = []

    for idx, item in enumerate(column_info_raw):
        if item[0] == 'C':
            continuous.append(idx)
            column_info.append((float(item[1]), float(item[2])))
        else:
            assert item[0] == 'D'
            discrete.append(idx)
            column_info.append(item[1:])

    meta = {
        'continuous_columns': continuous,
        'discrete_columns': discrete,
        'column_info': column_info,
    }

    with open(data_filename) as f:
        lines = f.readlines()

    data = []
    for row in lines:
        row_raw = row.split()
        row = []
        for idx, col in enumerate(row_raw):
            if idx in continuous:
                row.append(col)
            else:
                assert idx in discrete
                row.append(column_info[idx].index(col))

        data.append(row)

    return np.asarray(data, dtype='float32'), meta['discrete_columns']


def write_tsv(data, meta, output_filename):
    """Write to a tsv file."""
    with open(output_filename, 'w') as f:
        for row in data:
            for idx, col in enumerate(row):
                if idx in meta['continuous_columns']:
                    print(col, end=' ', file=f)
                else:
                    assert idx in meta['discrete_columns']
                    print(meta['column_info'][idx][int(col)], end=' ', file=f)

            print(file=f)

def get_null_mask(df):
    """
    Retorna uma máscara com 1 onde o DataFrame tem valores nulos,
    e 0 onde o valor está presente.
    """
    null_mask = df.isna().astype(int)
    return null_mask

def generate_incomplete_data(data, name, saving_dir, percentage = 0.10, np_seed = 42, columns = ['Vazao', 'Vazao_bbr']):
    """
    Returns and saves a dataset with missing data
    """

    # Defining a seed for replication
    np.random.seed(np_seed)

    # Specifiing the missing data percentage in the name of diretory and creating if does not exist
    percentage_string = str(int(percentage * 100))
    saving_dir = f"{saving_dir}_{percentage_string}"
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
        print(f"'{saving_dir}' directory was created for saving incomplete_data")

    incomplete_data = data.copy()
    for column in columns:
        col = data.columns.get_loc(column)

        n_test = data.shape[0]
        num_missing = int(n_test * percentage)

        missing_rows = np.random.choice(n_test, num_missing, replace=False)

        incomplete_data.iloc[missing_rows, col] = np.nan

    incomplete_data.to_csv(f"{saving_dir}/{name}")

    return incomplete_data
