import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from functools import partial
from joblib import Parallel, delayed


if __name__ == "__main__":
    # Set paths
    path_dfs_base = os.path.join(os.path.dirname(os.getcwd()), 'data')
    df_meta = pd.read_csv(os.path.join(path_dfs_base, 'df_meta.csv'))
    path_df_tile = os.path.join(path_dfs_base, 'df_tile.pkl')
    
    # Set base paths
    path_dfs_in_base = os.path.join(path_dfs_base, 'predictions_test/df_pred_{}_{}.pkl')
    path_df_out = os.path.join(path_dfs_base, 'df_preds_test.pkl')

    # Set cluster_ids and folds to loop over
    clusters = np.arange(50)
    n_folds = 10

    # Make parallelisable function to collect predictions
    def collect_cluster_preds(cluster_id, n_folds, path_dfs_in_base):
        cols_curr = None
        case_ids = None
        folds = np.arange(n_folds)

        # Get case ids, genes of this cluster etc.
        path_curr = path_dfs_in_base.format(cluster_id, 0)
        df_tile_preds = pd.read_pickle(path_curr)

        cols_curr = [col for col in df_tile_preds.columns if col.startswith('ENS')]
        cols_curr_mean = [col + '_mean' for col in cols_curr]
        cols_curr_weighted = [col + '_weighted_average' for col in cols_curr]
        cols_out = cols_curr_mean + cols_curr_weighted

        case_ids = df_tile_preds['case_id'].unique()

        vals_all = np.zeros((len(case_ids), len(cols_out)))

        # Loop over case ids
        for i, case_id in enumerate(case_ids):

            #Loop over folds
            for fold in folds:
                path_curr = path_dfs_in_base.format(cluster_id, fold)
                df_tile_preds = pd.read_pickle(path_curr)

                vals_case_fold = df_tile_preds[cols_curr].loc[df_tile_preds['case_id'] == case_id].values

                if fold == 0:
                    vals_case = np.zeros((*vals_case_fold.shape, n_folds))

                vals_case[..., fold] = vals_case_fold

            # Get mean and variance per tile
            vars_case = np.var(vals_case, axis=2)
            means_case = np.mean(vals_case, axis=2)

            # Get weights for inverse variance weighting
            weights = 1 / vars_case / np.sum(1 / vars_case, axis=0)
            
            # Write to output array
            vals_all[i, 0:len(cols_curr)] = np.mean(means_case, axis=0)
            vals_all[i, len(cols_curr):2*len(cols_curr)] = np.average(means_case, weights=weights, axis=0)

        # Make df, return
        df_out = pd.DataFrame(data=vals_all, index=case_ids, columns=cols_out)
        return df_out


    # Loop over all clusters and folds, collect predictions, 
    # get mean and median per case
    # Set up partial function
    collect_cluster_preds_par = partial(collect_cluster_preds, 
                                        n_folds=n_folds, 
                                        path_dfs_in_base=path_dfs_in_base)

    # Collect clusters in parallel
    df_list_clusts = Parallel(n_jobs=4)(delayed(collect_cluster_preds_par)(cluster_id) 
                                                for cluster_id in tqdm(clusters))

    # Concatenate all clusters into one df
    df_preds = pd.concat(df_list_clusts, axis=1)
    df_preds.to_pickle(path_df_out)
            