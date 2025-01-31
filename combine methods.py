# %% 
import os
import pickle
from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
import netbone as nb
from concurrent.futures import ThreadPoolExecutor, as_completed

# %%
# Get a list of all CSV files in the data directory
csv_files = [f for f in os.listdir('Data') if f.endswith('.csv')]

# %%
def edges(row):
    if row.source < row.target:
        return row.source + '-' + row.target
    return row.target + '-' + row.source

# %%
for csv_file in csv_files:
    try:
        with open('Results/MSP/' + csv_file.split('.')[0] + '.pickle', 'rb') as f:
            backbone = pickle.load(f)
        msp_df = backbone.to_dataframe()
        msp_df['edges'] = msp_df.apply(lambda row: edges(row), axis=1)
        # msp_df = msp_df.drop(columns=['source', 'target', 'weight'])

        with open('Results/PLAM/' + csv_file.split('.')[0] + '.pickle', 'rb') as f:
            backbone = pickle.load(f)
        plam_df = backbone.to_dataframe()
        plam_df['edges'] = plam_df.apply(lambda row: edges(row), axis=1)
        plam_df = plam_df.drop(columns=['source', 'target', 'weight'])

        with open('Results/PMFG/' + csv_file.split('.')[0] + '.pickle', 'rb') as f:
                backbone = pickle.load(f)
        pmfg_df = backbone.to_dataframe()
        pmfg_df['edges'] = pmfg_df.apply(lambda row: edges(row), axis=1)
        pmfg_df = pmfg_df.drop(columns=['source', 'target', 'weight'])

        with open('Results/Metric Distance Backbone/' + csv_file.split('.')[0] + '.pickle', 'rb') as f:
                backbone = pickle.load(f)
        mb_df = backbone.to_dataframe()
        mb_df['edges'] = mb_df.apply(lambda row: edges(row), axis=1)
        mb_df = mb_df.drop(columns=['source', 'target', 'weight', 'distance'])

        with open('Results/Ultrametric Distance Backbone/' + csv_file.split('.')[0] + '.pickle', 'rb') as f:
                backbone = pickle.load(f)
        umb_df = backbone.to_dataframe()
        umb_df['edges'] = umb_df.apply(lambda row: edges(row), axis=1)
        umb_df = umb_df.drop(columns=['source', 'target', 'weight', 'distance'])

        with open('Results/H-Backbone/' + csv_file.split('.')[0] + '.pickle', 'rb') as f:
                backbone = pickle.load(f)
        hb_df = backbone.to_dataframe()
        hb_df['edges'] = hb_df.apply(lambda row: edges(row), axis=1)
        hb_df = hb_df.drop(columns=['source', 'target', 'weight', 'h_bridge'])

        with open('Results/High Salience Skeleton/' + csv_file.split('.')[0] + '.pickle', 'rb') as f:
                backbone = pickle.load(f)
        hss_df = backbone.to_dataframe()
        hss_df['edges'] = hss_df.apply(lambda row: edges(row), axis=1)
        hss_df = hss_df.drop(columns=['source', 'target', 'weight', 'distance', 'salience'])

        with open('Results/Doubly Stochastic/' + csv_file.split('.')[0] + '.pickle', 'rb') as f:
                backbone = pickle.load(f)
        ds_df = backbone.to_dataframe()
        ds_df['edges'] = ds_df.apply(lambda row: edges(row), axis=1)
        ds_df = ds_df.drop(columns=['score', 'source', 'target', 'weight'])

        merged_df = pd.merge(plam_df, pmfg_df, on='edges', suffixes=['_PLAM','_PMFG'])
        merged_df = pd.merge(merged_df, mb_df, on='edges')
        merged_df = pd.merge(merged_df, umb_df, on='edges', suffixes=['_MB','_UMB'])
        merged_df = pd.merge(merged_df, hb_df, on='edges')
        merged_df = pd.merge(merged_df, hss_df, on='edges', suffixes=['_HB', '_HSS'])
        merged_df = pd.merge(merged_df, msp_df,  on='edges')
        merged_df = pd.merge(merged_df, ds_df, how='left', on='edges', suffixes=['_MSP','_DS'])
        merged_df['in_backbone_DS'] = merged_df['in_backbone_DS'].fillna(False)

        os.makedirs('Results/Combined', exist_ok=True)
        with open('Results/Combined/' + csv_file.split('.')[0] + '.pickle', 'wb') as f:
            pickle.dump(merged_df, f, pickle.HIGHEST_PROTOCOL)
    except:
        print(csv_file)
        continue


# %%
