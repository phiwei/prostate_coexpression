import os
import gzip
import argparse
import numpy as np 
import pandas as pd 

from tqdm import tqdm
from pathlib import Path


# Define RNA collecting function
def get_rna_df(path, case_id):
    lines = list()
    with gzip.open(path,'rt') as f:
        for line in f:
            line_tmp = line[:-2]
            if line_tmp.startswith('ENS'):
                lines.append(line_tmp)

    a = np.asarray([sub.split('\t') for sub in lines])
    columns = a[:, 0]
    columns = [col.split('.')[0] for col in columns]

    data = list()
    for entry in a[:, 1]:
        if len(entry) == 0:
            data.append(0)
        else:
            if float(entry).is_integer():
                try:
                    data.append(int(entry))
                except:
                    data.append(float(entry))
            else:
                data.append(float(entry))

    data = np.asarray(data)
    data = np.expand_dims(data, axis=0)

    df = pd.DataFrame(data=data, columns=columns)
    df['case_id'] = case_id
    df = df.set_index('case_id', drop=True)
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default=None, help='Path to TCGA RNA data.')
    args = parser.parse_args()
    path_data_base = args.indir

    # Set paths
    path_dfs_base = os.path.join(os.path.dirname(os.getcwd()), 'data')
    df_meta = pd.read_csv(os.path.join(path_dfs_base, 'df_meta.csv'))
    df_clusters = pd.read_csv(os.path.join(path_dfs_base, 'df_clusters.csv'))
    path_df_rna_out = os.path.join(path_dfs_base, 'df_RNA.pkl')

    # Create a lookup table for all files
    paths = list()
    names = list()

    for path in Path(path_data_base).rglob('*.gz'):
        name = str(path.name)
        if name.startswith('._'):
            continue
        paths.append(str(path))
        names.append(str(path.name))
    dict_lookup = dict(zip(names, paths))
    dict_lookup;

    # Iterate over all cases, collect their RNA and merge into df with all RNA information
    dfs = list()
    print('Collecting RNA-seq data...')
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        case_id = row['case_id']
        filename = row['rna_name']
        filepath = dict_lookup[filename]
        dfs.append(get_rna_df(filepath, case_id))

    # Merge to df_rna, select genes, save as pickle
    df_rna = pd.concat(dfs)
    df_rna = df_rna[df_clusters['ensemble_id'].tolist()]
    df_rna = df_rna.applymap(lambda x: np.log2(1 + x))

    # Save df
    df_rna.to_pickle(path_df_rna_out)