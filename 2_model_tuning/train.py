if __name__ == '__main__':
    import os
    import sys
    import argparse
    import datetime
    import numpy as np

    from ray import tune

    sys.path.append(os.getcwd())
    from train_utils import RNAPrediction

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, help='Path to tiles.')
    parser.add_argument('--modeldir', type=str, help='Path to model checkpoints.')
    parser.add_argument('--n_workers', type=int, default=-1, help='Number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size per GPU.')
    parser.add_argument('--mixed', type=bool, default=False, help='Whether to use mixed precision training.')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='Whether to use multi-gpu training.')


    args = parser.parse_args()
    path_data_base = args.indir
    path_out_base = args.modeldir
    n_workers = args.n_workers
    batch_size = args.batch_size
    mixed = args.mixed
    multi_gpu = args.multi_gpu

    time_str = 'models-{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    path_out_base = os.path.join(path_out_base, time_str)

    # General setup
    config = dict()
    config['task'] = 'regression'
    config['validate'] = False
    config['verbose'] = True
    config['n_workers'] = n_workers
    config['path_col'] = 'path_col'
    config['strat_col'] = 'slide_filename'
    config['slide_col'] = 'slide_filename'

    # Write paths into config
    config['path_data_base'] = path_data_base
    config['path_dfs_base'] = os.path.join(os.path.dirname(os.getcwd()), 'data')

    # Model setup
    config['model'] = 'resnet'
    config['pretrained'] = True
    config['multi_gpu'] = multi_gpu
    config['mixed'] = mixed
    config['p_dropout'] = 0.5

    # Image setup
    config['img_shape'] = 440

    # Dataloader setup
    config['replacement'] = True
    config['sample_weights'] = 'log_count'
    config['batch_size'] = batch_size

    # Train setup
    config['lr'] = 1e-3
    config['weight_decay'] = 0
    config['scheduler'] = 'ExponentialLR'
    config['gamma'] = 0.9
    config['num_samples_train'] = 24000
    config['num_samples_valid'] = 6400
    config['n_epochs'] = 25
    
    # Iterate over clusters and folds
    config['cluster_id'] = tune.grid_search(np.arange(50).tolist())
    config['fold'] = tune.grid_search(np.arange(10).tolist())

    # Run tuning
    name = os.path.basename(__file__)[:-3]

    tune.run(
        RNAPrediction,
        name=name, 
        config=config,
        stop={'training_iteration': config['n_epochs']},
        local_dir=path_out_base, 
        checkpoint_freq=0, 
        checkpoint_at_end=True,
        resources_per_trial={'cpu': config['n_workers'], 'gpu': 1}, 
        num_samples=1)
