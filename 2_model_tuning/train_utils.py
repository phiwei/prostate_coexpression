import os
import sys
import glob
import time
import torch
import platform
import numpy as np
import pandas as pd
import torch.nn as nn

from ray import tune
from tqdm import tqdm
from scipy import stats
from copy import deepcopy
from functools import partial
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_sample_weight

if platform.system() == 'Linux':
    from accimage import Image
else:
    from PIL import Image


def get_dataloader(config, batch_size, steps_per_epoch=None, 
                   num_samples=None, 
                   df_tile=None, df_slides=None, 
                   label_cols=None, transform=None):

    if label_cols is None:
        label_cols = config['label_cols']
    path_data_base = config['path_data_base']
    slide_col = config['slide_col']
    strat_col = config.get('strat_col', None)
    n_workers = config.get('n_workers', 0)
    sample_weights = config.get('sample_weights', None)
    replacement = config.get('replacement', True)
    if not replacement:
        print('Warning: replacement=False is ignored for this dataloader.')

    assert (steps_per_epoch is None) ^ (num_samples is None)

    if df_slides is None:
        # Set up small df where each slide only has one row with labels
        df_slides = df_tile.drop_duplicates(subset=slide_col).reset_index(drop=True)

        # Count how many tiles there are per slide
        tile_numbers = list()
        for slide in df_slides[slide_col].values:
            idx_bool = df_tile[slide_col] == slide
            tile_numbers.append(np.sum(idx_bool))
        tile_numbers = np.asarray(tile_numbers)
    else:
        tile_numbers = df_slides['n_tiles'].values
    
    # This probably needs to be changes for this mode of sampling
    if strat_col is not None:
        stratifier = df_slides[strat_col].values

    # Generate slide paths and data set
    labels = df_slides[label_cols].values
    slide_paths = [os.path.join(path_data_base, slide) for slide in df_slides[slide_col].values]
    dataset = Dataset(slide_paths=slide_paths, tile_numbers=tile_numbers, 
                      labels=labels, transform=transform)

    if num_samples is None:
        num_samples = steps_per_epoch*batch_size

    # Compute sample weights depending on balancing mode. 
    if sample_weights == 'balanced':
        # number of batches / samples will be determined by
        # 2* minority class or steps_per_epoch, whatever is less
        
        # Pick really large number to make it irrelevant
        if steps_per_epoch is None:
            steps_per_epoch = sys.maxsize

        weights = compute_sample_weight(class_weight='balanced',
                                        y=stratifier)
        num_samples = np.min((steps_per_epoch*batch_size,
                              int(np.min(np.bincount(stratifier.astype(int))) * config['n_labels'])))

        sampler = torch.utils.data.WeightedRandomSampler(weights, 
                                                         num_samples=int(num_samples), 
                                                         replacement=True)
    elif sample_weights == 'log_count':
        # Something in between balanced and random sampling, 
        # idea is to make sampling frequency dependent on log()
        # of number of tiles of a patient to sample patients with
        # more samples more often but in a more balanced way. 
        weights = np.log(tile_numbers + 1)
        sampler = torch.utils.data.WeightedRandomSampler(weights, 
                                                         num_samples=int(num_samples), 
                                                         replacement=True)
    else:
        weights = None
        sampler = None

    if weights is not None:
        assert not sum(weights == 0), 'Some samples have zero weight.'
        assert not sum(np.isnan(weights)), 'Some sample weights are NaN.'

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler,
                            num_workers=n_workers,
                            pin_memory=True)

    return dataloader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, slide_paths, tile_numbers, 
                labels, transform=None, img_format='jpg'):

        self.slide_paths = slide_paths
        self.tile_numbers = tile_numbers
        self.labels = labels
        self.transform = transform
        self.format = img_format

        if platform.system() == 'Linux':
            self.load = load_acc
        else:
            self.load = load_PIL


    def __len__(self):
        return len(self.slide_paths)

    def __getitem__(self, index):
        # Generate the tile path using the slide 
        # folder path and a randomly drawn integer
        path_slide = self.slide_paths[index]
        tile_paths = glob.glob(os.path.join(path_slide, '*.' + self.format))

        path_img = np.random.choice(tile_paths)
        assert os.path.isfile(path_img), 'Image path does not exist: {}'.format(path_img)
        image = self.load(path_img)
        target = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def load_acc(path_img):
    return Image(path_img)

def load_PIL(path_img):
    return Image.open(path_img)


def append_normalised_RNA_labels(df_tile_train, df_tile_valid, df_RNA, df_cluster_ids, cluster_id):  
    # Get genes of current cluster
    if cluster_id == 'all':
        genes = df_cluster_ids['ensemble_id'].values
    else:
        genes = df_cluster_ids['ensemble_id'].loc[df_cluster_ids['cluster_id'] == cluster_id].values
    df_RNA_cluster = df_RNA[genes]

    # Normalise df_RNA_cluster using only training data samples to compute mean and variance
    print()
    print('Normalising genes...')
    print()
    ids_train = set(df_tile_train['case_id'].unique())
    means = list()
    vars = list()
    for gene in tqdm(genes):
        gene_vals = df_RNA_cluster[gene].values
        gene_vals_train = df_RNA_cluster[gene].loc[df_RNA_cluster.index.map(lambda x: x in ids_train)].values
        mean_train = np.mean(gene_vals_train)
        var_train = np.var(gene_vals_train)
        df_RNA_cluster[gene] = (gene_vals - mean_train) / var_train
        means.append(mean_train)
        vars.append(var_train)
    means = np.asarray(means)
    vars = np.asarray(vars)

    print()
    print('Adding cluster RNA labels to df_tile_train...')
    print()
    rna_tmp = np.zeros((len(df_tile_train), len(genes)))
    for case_id in tqdm(df_tile_train['case_id'].unique()):
        idx_id = df_tile_train['case_id'] == case_id
        repeats = sum(idx_id)
        rna_tmp[idx_id, ...] = np.repeat(df_RNA_cluster.loc[case_id].values[np.newaxis, ...], 
                                            repeats=repeats, axis=0)

    df_tmp = pd.DataFrame(rna_tmp, columns=genes)
    df_tile_train = pd.concat([df_tile_train, df_tmp], axis=1)

    if df_tile_valid is not None:
        print()
        print('Adding cluster RNA labels to df_tile_valid...')
        print()
        rna_tmp = np.zeros((len(df_tile_valid), len(genes)))
        for case_id in tqdm(df_tile_valid['case_id'].unique()):
            idx_id = df_tile_valid['case_id'] == case_id
            repeats = sum(idx_id)
            rna_tmp[idx_id, ...] = np.repeat(df_RNA_cluster.loc[case_id].values[np.newaxis, ...], 
                                                repeats=repeats, axis=0)

        df_tmp = pd.DataFrame(rna_tmp, columns=genes)
        df_tile_valid = pd.concat([df_tile_valid, df_tmp], axis=1)
        
    return df_tile_train, df_tile_valid, genes, means, vars


class Trainer(tune.Trainable):
    def _internal_setup(self, means, vars):
        # Detect if we have a GPU available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = initialize_model(self.config).to(self.device)
        self.model.means = means
        self.model.vars = vars
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                   lr=self.config['lr'], 
                                   momentum=self.config.get('momentum', 0.9),
                                   nesterov=True)
        if 'lr_patience' in self.config and self.config['scheduler'] == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.2, 
                patience=self.config.get('lr_patience', sys.maxsize))
        elif self.config['scheduler'] == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                step_size=self.config['lr_step_size'], gamma=0.2)
        elif self.config['scheduler'] == 'ExponentialLR':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 
                                                              gamma=self.config.get('gamma', 0.9))
        else:
            self.scheduler = None

        # Set scaler if using mixed precision
        if self.config.get('mixed', False):
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler=None

        # If metric not defined, set it to None
        try:
            _ = self.metrics
        except:
            self.metrics = None

        # Set up some internal variables
        self.epoch = 0
        
        # Set up early stopping or set numbers
        # that circumvent it if patience is not set
        self.stop_early = False
        self.idx_early = 0
        self.best_epoch = 0
        self.patience = self.config.get('patience', float('inf'))
        if self.patience == float('inf'):
            self.best_loss = 0
        else:
            self.best_loss = float('inf')
        
        self.validate = self.config.get('validate', True)   
        if not self.validate:
            self.data_valid = None   

    def _train(self):

        data_log = dict()

        # First, train model for an epoch
        data_log = self._execute_phase(model=self.model, dataloader=self.data_train, 
                                       criterion=self.criterion, optimizer=self.optimizer,
                                       device=self.device, config=self.config, epoch=self.epoch, 
                                       phase='train', data_log=data_log, 
                                       scheduler=self.scheduler, scaler=self.scaler)
        

        # Potentially validate model
        if self.validate:
            data_log = self._execute_phase(model=self.model, dataloader=self.data_valid, 
                                           criterion=self.criterion, optimizer=self.optimizer,
                                           device=self.device, config=self.config, epoch=self.epoch, 
                                           phase='val', data_log=data_log, metrics=self.metrics, 
                                           var_names=self.var_names, 
                                           scheduler=self.scheduler, scaler=self.scaler)

            # Early stopping
            # ... update early stopping and saving model weights
            epoch_loss = data_log['val_loss']
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.idx_early = 0
                self.best_epoch = deepcopy(self.epoch)
                data_log['stop_flag'] = False

                # Save model
                print('Saving model at epoch', self.epoch)
                torch.save(self.model, './model.pth')

            elif epoch_loss > self.best_loss:
                self.idx_early += 1
                data_log['stop_flag'] = False
                if self.idx_early >= self.patience:
                    data_log['stop_flag'] = True
        else:
            data_log['stop_flag'] = False

        data_log['best_epoch'] = self.best_epoch
        self.epoch += 1

        return data_log

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model, checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


def initialize_model(config):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.

    model_name = config['model']
    n_labels = config['n_labels']
    task = config['task']
    pretrained = config.get('pretrained', True)
    multi_gpu = config.get('multi_gpu', False)
    p_dropout = config.get('p_dropout', 0.)

    def _get_final_layer(task, n_labels, n_feats, p_dropout):
        if task == 'classification'and n_labels == 1:
            layer_out = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p_dropout),
                nn.Linear(n_feats, n_labels),
                nn.Sigmoid())
        else:
            layer_out = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p_dropout),
                nn.Linear(n_feats, n_labels))

        return layer_out

    model = None

    if model_name == "resnet":
        model = models.resnet18(pretrained=pretrained)
        n_feats = model.fc.in_features
        model.fc = _get_final_layer(task, n_labels, n_feats, p_dropout)

    elif model_name == "inception":
        model = models.inception_v3(pretrained=pretrained)
        n_feats = model.fc.in_features
        model.fc = _get_final_layer(task, n_labels, n_feats, p_dropout)

    print('Detected GPUs:', torch.cuda.device_count())
    if multi_gpu:
        print()
        print('Using multiple GPUs:', torch.cuda.device_count())
        print()
        model = nn.DataParallel(model)

    return model


def spearman_score(y_true, y_pred):
    assert not np.isnan(y_true).any()
    assert not np.isinf(y_true).any()
    assert not np.isnan(y_pred).any()
    assert not np.isinf(y_pred).any()
    corrs = list()
    for idx_col in range(y_true.shape[1]):
        corr = stats.spearmanr(y_true[:, idx_col], y_pred[:, idx_col]).correlation
        corrs.append(corr)
    return np.asarray(corrs)


def execute_phase(model, dataloader, criterion, optimizer,
                  device, config, epoch, phase, data_log, var_names=None,
                  metrics=None, scheduler=None, scaler=None):
    # Get config
    n_epochs = config.get('n_epochs', 50)
    verbose = config.get('verbose', True)
    mixed = config.get('mixed', False)

    # Set up depending on phase
    if phase == 'train':
        # Set model to training mode
        model.train() 
        data_log['lr'] = optimizer.param_groups[0]['lr']

        if verbose:
            print('Epoch: %d/%d' % (epoch + 1, n_epochs))
    else:
        model.eval()

    # Some variables
    running_loss = 0.0
    running_time = 0.0
    idx_batch = 0
    y_trues = list()
    preds = list()

    # Iterate over data
    for inputs, labels in tqdm(dataloader):

        # Some logging
        time_batch_start = time.time()
        if phase == 'val':
            y_true = labels.numpy().copy()
            y_trues.append(y_true)

        # Send data to device
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Pass batch through model and backprop
        with torch.set_grad_enabled(phase == 'train'):
            
            # With mixed precision
            if mixed:
                with torch.cuda.amp.autocast():
                    if (config.get('model', '') == 'inception') and (phase == 'train'):
                        outputs, _ = model(inputs)
                    else:
                        outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            # Without mixed precision
            else:

                if (config.get('model', '') == 'inception') and (phase == 'train'):
                    outputs, _ = model(inputs)
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

        # Keep predictions for metric calculation
        if phase == 'val':
            preds.append(outputs.detach().cpu().numpy())

        # Keep some info
        running_loss += loss.item()
        running_time += (time.time() - time_batch_start) / inputs.size(0)
        idx_batch += 1

    # Write stuff to data log
    epoch_loss = running_loss / (idx_batch)
    time_per_img = running_time / (idx_batch) * 1000
    data_log[phase + '_loss'] = epoch_loss
    data_log[phase + '_ms_per_img'] = time_per_img

    # Compute metrics for validation data only
    if phase == 'val' and metrics is not None:
        y_trues = np.concatenate(y_trues)
        preds = np.concatenate(preds)
        metric_log = dict()
        # Collect metric values
        for key in metrics:
            metric_vals = metrics[key](y_trues, preds)
            metric_names = [key + '_' + name for name in var_names]
            metric_log.update(zip(metric_names, metric_vals))

        data_log.update(metric_log)

    # Update lr scheduler
    if scheduler is not None:
        if config['validate'] and phase == 'val':
            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(epoch_loss)
            else:
                scheduler.step()
        elif config['validate'] and phase != 'val':
            pass
        else:
            scheduler.step()

    return data_log



class RNAPrediction(Trainer):
    def _setup(self, config):
        # Load df_tile, RNA data and cluster df
        path_dfs_base = config['path_dfs_base']
        path_df_tile = os.path.join(path_dfs_base, 'df_tile.pkl')
        path_df_clusters = os.path.join(path_dfs_base, 'df_clusters.csv')
        path_df_RNA = os.path.join(path_dfs_base, 'df_RNA.pkl')
        df_cluster_ids = pd.read_csv(path_df_clusters)
        df_RNA = pd.read_pickle(path_df_RNA)
        df_tile = pd.read_pickle(path_df_tile)

        config['n_epochs'] = 25

        # Drop test data, drop validation data
        df_tile_dev = df_tile.loc[~df_tile['Test'].astype(bool)].reset_index(drop=True)
        df_tile_dev = df_tile_dev.loc[df_tile_dev['CV'] != config['fold']].reset_index(drop=True)

        # Swith to df_slides from here on
        df_slides = df_tile_dev.drop_duplicates(subset='case_id').reset_index(drop=True)

        # Count how many tiles there are per slide
        tile_numbers = list()
        for case in df_slides['case_id'].values:
            idx_bool = df_tile_dev['case_id'] == case
            if np.sum(idx_bool) == 0:
                print('Slide {} has 0 tiles and will be removed.'.format(case))
            tile_numbers.append(np.sum(idx_bool))
        df_slides['n_tiles'] = np.asarray(tile_numbers)
        df_slides_train = df_slides
        df_slides_valid = None
        
        # Append RNA data to df_tile, save var_names for metric names
        df_slides_train, df_slides_valid, label_cols, means_train, vars_train = append_normalised_RNA_labels(df_slides_train, df_slides_valid,  
                                                                                    df_RNA, df_cluster_ids, 
                                                                                    config['cluster_id'])
        self.var_names = label_cols
        self.config['n_labels'] = len(label_cols)
        
        # Set up transform for images with augmentations and normalisations
        trans_train = transforms.Compose([transforms.RandomCrop(config['img_shape']),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                                            ])

        # Create training and validation datasets, clear tile dfs to free RAM
        print('Creating dataloaders...')
        data_train = get_dataloader(df_slides=df_slides_train, config=config, 
                                    label_cols=self.var_names,
                                    num_samples=config['num_samples_train'],
                                    batch_size=config['batch_size'], 
                                    transform=trans_train)
        self.data_train = data_train

        del df_tile, df_tile_dev

        for inputs, labels in data_train:
            print('Shapes of inputs and labels:')
            print(inputs.shape)
            print(labels.shape)
            break

        # Setup loss and metrics
        if config.get('experimental_loss'):
            reduction = 'none'
        else:
            reduction = 'mean'
        self.criterion = nn.MSELoss(reduction=reduction)

        self.metrics = {'spearman': spearman_score, 
                        'r2': partial(r2_score, 
                                      multioutput='raw_values'), 
                        'mse': partial(mean_squared_error, 
                                       multioutput='raw_values')}

        # Initialize model, optimizer, scheduler etc.
        self._internal_setup(means_train, vars_train)
        self._execute_phase = execute_phase

