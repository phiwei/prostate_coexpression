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
    path_dfs_in_base = os.path.join(path_dfs_base, 'predictions_valid/df_pred_{}_{}.pkl')
    path_df_out = os.path.join(path_dfs_base, 'df_preds_valid.pkl')

    # Set cluster_ids and folds to loop over
    clusters = np.arange(50)
    n_folds = 10

    # Make parallelisable function to collect predictions
    def collect_cluster_preds(cluster_id, n_folds, path_dfs_in_base):
        df_list_folds = list()
        cols_curr = None
        folds = np.arange(n_folds)

        #Loop over folds
        for fold in folds:

            # Set path, load df
            path_curr = path_dfs_in_base.format(cluster_id, fold)
            df_tile_preds = pd.read_pickle(path_curr)

            # If first fold, get prediction cols of this cluster
            if cols_curr is None:
                cols_curr = [col for col in df_tile_preds.columns if col.startswith('ENS')]
                cols_curr_mean = [col + '_mean' for col in cols_curr]
                cols_curr_median = [col + '_median' for col in cols_curr]
                cols_out = cols_curr_mean + cols_curr_median

            # Iterate over case ids, get mean and median of predictions per case
            case_ids = list()
            vals_all = np.zeros((len(set(df_tile_preds['case_id'])), 2*len(cols_curr)))
            for i, case_id in enumerate(set(df_tile_preds['case_id'])):

                # Get predictions for current caste id
                vals = df_tile_preds[cols_curr].loc[df_tile_preds['case_id'] == case_id].values
                case_ids.append(case_id)

                # Get mean and median of predictions
                vals_all[i, 0:len(cols_curr)] = np.mean(vals, axis=0)
                vals_all[i, len(cols_curr):2*len(cols_curr)] = np.median(vals, axis=0)

            # Create dataframe for this fold
            df_tmp = pd.DataFrame(data=vals_all, index=case_ids, columns=cols_out)
            df_list_folds.append(df_tmp)

        # Concatenate all folds into one df
        df_preds = pd.concat(df_list_folds, axis=0)
        
        return df_preds


    # Loop over all clusters and folds, collect predictions, 
    # get mean and median per case
    # Set up partial function
    collect_cluster_preds_par = partial(collect_cluster_preds, 
                                        n_folds=n_folds, 
                                        path_dfs_in_base=path_dfs_in_base)

    # Collect clusters in parallel
    df_list_clusts = Parallel(n_jobs=-1)(delayed(collect_cluster_preds_par)(cluster_id) 
                                         for cluster_id in tqdm(clusters))

    # Concatenate all clusters into one df
    df_preds = pd.concat(df_list_clusts, axis=1)
    df_preds.to_pickle(path_df_out)

