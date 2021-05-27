import os
import sys
import copy
import torch
import platform
import numpy as np
import pandas as pd

from ray import tune
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_sample_weight

if platform.system() == 'Linux':
    from accimage import Image
else:
    from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels,
                 transform=None):

        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

        assert len(file_paths) == len(labels)

        if platform.system() == 'Linux':
            self.load = load_acc
        else:
            self.load = load_PIL


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):

        path_img = self.file_paths[index]
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


def predict_rna(config):

    # Get paths
    path_dfs_base = config['path_dfs_base']
    path_data_base = config['path_data_base']
    path_models_base = config['path_models_base']

    # Make sure we are at correct subfolder level with model path
    if os.listdir(path_models_base)[0] == 'train':
        path_models_base = os.path.join(path_models_base, 'train')

    path_df_tile = os.path.join(path_dfs_base, 'df_tile.pkl')
    path_df_clusters = os.path.join(path_dfs_base, 'df_clusters.csv')

    print('Loading dataframes...')
    df_cluster_ids = pd.read_csv(path_df_clusters)
    df_tile = pd.read_pickle(path_df_tile)

    # Get genes
    genes = df_cluster_ids['ensemble_id'].loc[df_cluster_ids['cluster_id'] == config['cluster_id']].values

    # Drop non-cancer tiles, select validation or test data
    if config['set'] == 'valid':
        df_tile = df_tile.loc[~df_tile['Test'].astype(bool)].reset_index(drop=True)
        df_tile = df_tile.loc[df_tile['CV'] == config['fold']].reset_index(drop=True)
        
    elif config['set'] == 'test':
        df_tile = df_tile.loc[df_tile['Test'].astype(bool)].reset_index(drop=True)

    print('Final number of ids: ', len(set(df_tile['case_id'])))
    
    # Set up transform for images with augmentations and normalisations
    trans_valid = transforms.Compose([transforms.CenterCrop(config['img_shape']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]),
                                        ])

    # Create dataloader
    labels = np.arange(len(df_tile))
    filenames = [os.path.join(path_data_base, row['slide_filename'], row['tile_filename'])
                           for  _, row in tqdm(df_tile.iterrows(), total=len(df_tile))]
    
    dataset = Dataset(filenames, labels,
                      transform=trans_valid)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                            sampler=None,
                            num_workers=config['n_workers'],
                            pin_memory=True)

    df_out = df_tile.copy()
    df_out = df_out.drop(labels='tile_filename', axis=1)

    # Find model checkpoint path, load model
    search_str = '*cluster_id={},fold={}*/**/*.pth'.format(config['cluster_id'], config['fold'])
    device = 'cuda:0'
    path_model = str(list(Path(path_models_base).rglob(search_str))[0])
    model = torch.load(path_model, map_location=device)
    model.eval()

    # Collect all prediction in dataset
    predictions = list()
    labels = list()
    for images, labels_mock in tqdm(dataloader):
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.detach().cpu().numpy()

        predictions.append(copy.deepcopy(outputs))
        labels.append(copy.deepcopy(labels_mock))

    # Convert filenames from bytes to string, only take filename, not path
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    # Make sure that order is correct
    assert (labels == np.arange(len(labels))).all()

    # Collect predictions, scale with model train mean and variance
    for idx_pred, gene in enumerate(genes):
        df_out[gene] = predictions[:, idx_pred]*model.vars[idx_pred] + model.means[idx_pred]

    # Save as .pkl file
    path_df_preds = os.path.join(path_dfs_base, 
                                 'predictions_{}/df_pred_{}_{}.pkl'.format(config['set'], 
                                                                           config['cluster_id'], 
                                                                           config['fold']))
    df_out.to_pickle(path_df_preds)
    tune.report(complete=True)





        
