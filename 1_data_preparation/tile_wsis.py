import os
import argparse
import warnings
import openslide

import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from functools import partial
from joblib import Parallel, delayed


def tile(df_slide,
         case_id,
         path_img, 
         test, 
         cv, 
         path_out_base,
         tile_size,
         mpp_target):

    slide_filename = os.path.basename(path_img)
    slide_filename = os.path.splitext(slide_filename)[0]
    path_out = os.path.join(path_out_base, slide_filename)

    # Initialize the OpenSlide image
    slide = openslide.OpenSlide(path_img)
    mpp_source_base = float(slide.properties['openslide.mpp-x'])
    if slide.properties['openslide.mpp-x'] != slide.properties['openslide.mpp-y']:
        warnings.warn('Warning: Asymmetric pixel scaling...')
    scaling_base = mpp_target / mpp_source_base

    # Compute scaling at next-clostest resolution level
    scaling_seq = [x for x in slide.level_downsamples if x <= scaling_base]
    if scaling_seq:
        scaling_tmp = max(scaling_seq)
        resolution_level = slide.level_downsamples.index(scaling_tmp)
    else:
        resolution_level = 0
        scaling_tmp = slide.level_downsamples[resolution_level]
    mpp_source = mpp_source_base * slide.level_downsamples[resolution_level]
    scaling = mpp_target / mpp_source

    # Figure out the final tile size.
    tile_size_source = np.round(tile_size*scaling).astype(int)

    # Loop over all coordinates.
    data = list()
    for _, row in df_slide.iterrows():
        h_start = row['top']
        w_start = row['left']
        h_end = row['bottom']
        w_end = row['right']

        tile_filename = slide_filename + '_' + str(h_start) + '_' + str(h_end) \
                    + '_' + str(w_start) + '_' + str(w_end) + '.jpg'

        # Write tiles to disk
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        tile = slide.read_region(location=(w_start, h_start),
                                    level=resolution_level,
                                    size=(tile_size_source, tile_size_source))
        tile = tile.convert('RGB')

        if tile.width != tile_size or tile.height != tile_size:
            tile = tile.resize((tile_size, tile_size), Image.LANCZOS)
    
        tile.save(os.path.join(path_out, tile_filename), quality=80)

        # Store tile data.
        keys = ['case_id', 'slide_filename',  'tile_filename', 'top', 'bottom', 
                'left',  'right', 'mpp', 'mpp_source', 'lvl', 'Test', 'CV']
        values = [case_id, slide_filename, tile_filename, h_start, h_end, w_start, w_end, 
                  mpp_target, mpp_source, resolution_level, test, cv]

        data.append(dict(zip(keys, values)))

    # Make sure the OpenSlide image is closed.
    slide.close()

    # If there are entries in the data list, make a df and save it
    if not data:
        df_tmp = None
    else:
        df_tmp = pd.DataFrame(data)
        path_out_df = os.path.join(path_out, slide_filename + '.pkl')
        df_tmp.to_pickle(path_out_df)

    return df_tmp


if __name__ == '__main__':

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default=None, help='path/to/slides')
    parser.add_argument('--outdir', type=str, default=None, help='path/to/tiles')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel executions.')
    args = parser.parse_args()
    path_slides = args.indir
    path_tiles = args.outdir
    n_jobs  = args.n_jobs

    # Set paths, get dfs
    path_dfs_base = os.path.join(os.path.dirname(os.getcwd()), 'data')
    path_df_tile = os.path.join(path_dfs_base, 'df_tile.pkl')
    df_meta = pd.read_csv(os.path.join(path_dfs_base, 'df_meta.csv'))
    df_tile_coords = pd.read_pickle(os.path.join(path_dfs_base, 'df_tile_coordinates.pkl'))

    # Create a dictionary that maps all file names to paths
    paths = list()
    names = list()
    exists = 0
    for path in Path(path_slides).rglob('*.svs'):
        paths.append(str(path))
        names.append(str(path.name))
    print('Number of slides found:', len(paths))
    dict_lookup = dict(zip(names, paths))

    # Get lists for tiling
    l_dfs = list()
    l_ids = list()
    l_paths = list()
    l_test = list()
    l_cv = list()
    print('Collecting tiling data...')
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        case_id = row['case_id']
        slide_name = row['slide_name']
        test = row['test']
        cv = row['CV']
        df_tmp = df_tile_coords.loc[df_tile_coords['Case ID']==case_id].reset_index(drop=True)
        path_curr = dict_lookup[slide_name]
        
        l_dfs.append(df_tmp)
        l_ids.append(case_id)
        l_paths.append(path_curr)
        l_test.append(test)
        l_cv.append(cv)

    # Make partial tiling function for 40X
    tile_par = partial(tile, 
                       path_out_base=path_tiles, 
                       tile_size=500, 
                       mpp_target=0.252)

    print('Tiling WSIs...')
    dfs = Parallel(n_jobs=n_jobs)(delayed(tile_par)(df_slide, case_id, path_img, test, cv) 
                                  for df_slide, case_id, path_img, test, cv
                                  in tqdm(zip(l_dfs, l_ids, l_paths, l_test, l_cv), total=len(l_dfs)))

    # Concat output
    df_total = pd.concat(dfs).reset_index(drop=True)

    # Since all checks look ok, write to data
    df_total.to_pickle(path_df_tile)

