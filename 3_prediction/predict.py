if __name__ == '__main__':
    import os
    import sys
    import shutil
    import argparse
    import numpy as np

    from ray import tune

    sys.path.append(os.getcwd())
    from predict_utils import predict_rna

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default=None, help='Path to tiles.')
    parser.add_argument('--modeldir', type=str, default=None, help='Path to model checkpoints.')
    parser.add_argument('--n_workers', type=int, default=6, help='Number of dataloader workers.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size per GPU.')
    args = parser.parse_args()
    path_data_base = args.indir
    path_models_base = args.modeldir
    n_workers = args.n_workers
    batch_size = args.batch_size
    path_dfs_base = os.path.join(os.path.dirname(os.getcwd()), 'data')

    # General setup
    config = dict()
    config['n_workers'] = n_workers
    config['debug'] = False
    config['path_col'] = 'path_col'
    config['img_shape'] = 440
    config['balance'] = False
    config['shuffle'] = False
    config['model'] = 'resnet'
    config['multi_gpu'] = False
    config['mixed'] = True
    config['set'] = tune.grid_search(['valid', 'test'])
    config['batch_size'] = batch_size
    
    # Iterate over clusters and folds
    config['cluster_id'] = tune.grid_search(np.arange(50).tolist())
    config['fold'] = tune.grid_search(np.arange(10).tolist())

    # Save paths
    config['path_data_base'] = path_data_base
    config['path_models_base'] = path_models_base
    config['path_dfs_base'] = path_dfs_base

    # Make output paths
    paths_out = [os.path.join(path_dfs_base, 'predictions_valid'), 
                 os.path.join(path_dfs_base, 'predictions_test')]
    for path in paths_out:
        if not os.path.exists(path):
            os.makedirs(path)

    # Run tuning
    name = os.path.basename(__file__)[:-3]

    tune.run(predict_rna,
             name=name, 
             config=config,
             local_dir=os.path.join(path_dfs_base, 'tmp'), 
             resources_per_trial={'cpu': max(1, config['n_workers']), 'gpu': 1})
    shutil.rmtree(os.path.join(path_dfs_base, 'tmp'))

