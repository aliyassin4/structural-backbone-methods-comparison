# %% 
import os
import pickle
from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
import netbone as nb
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from collections import Counter

# %%
# Get a list of all CSV files in the data directory
csv_files = [f for f in os.listdir('Data') if f.endswith('.csv')]
# csv_files = ['lesmis.csv']

methods_dict = {nb.maximum_spanning_tree:'MSP',
                nb.plam:"PLAM", 
                nb.pmfg:"PMFG", 
                nb.doubly_stochastic:"Doubly Stochastic", 
                nb.h_backbone:"H-Backbone", 
                nb.metric_distance_backbone:"Metric Distance Backbone", 
                nb.ultrametric_distance_backbone:"Ultrametric Distance Backbone", 
                nb.high_salience_skeleton:"High Salience Skeleton"}

set_labels = ('MSP', 'PLAM', 'PMFG', "MB", 'UMB', 'HB', 'HSS', 'DS')
# %% 
backbones = {}
originals = {}
for csv_file in csv_files:
    try:
        with open('Results/Combined/' + csv_file.split('.')[0] + '.pickle', 'rb') as f:
            data = pickle.load(f)
        msp_df = data[data['in_backbone_MSP']][['source', 'target', 'edges', 'weight']]
        plam_df = data[data['in_backbone_PLAM']][['source', 'target', 'edges', 'weight']]
        pmfg_df = data[data['in_backbone_PMFG']][['source', 'target', 'edges', 'weight']]
        mb_df = data[data['in_backbone_MB']][['source', 'target', 'edges', 'weight']]
        umb_df = data[data['in_backbone_UMB']][['source', 'target', 'edges', 'weight']]
        hb_df = data[data['in_backbone_HB']][['source', 'target', 'edges', 'weight']]
        ds_df = data[data['in_backbone_DS']][['source', 'target', 'edges', 'weight']]
        hss_df = data[data['in_backbone_HSS']][['source', 'target', 'edges', 'weight']]
        edgelists = [msp_df, plam_df, pmfg_df, mb_df, umb_df, hb_df, hss_df, ds_df]
        backbones[csv_file.split('.')[0]] = [nx.from_pandas_edgelist(edgelist, edge_attr='weight') for edgelist in edgelists]
        originals[csv_file.split('.')[0]] = nx.read_weighted_edgelist(os.path.join('Data', csv_file))
    except Exception as e:
        print(csv_file, e)
        continue


# %%
def get_weights(G):
    return list(nx.get_edge_attributes(G, 'weight').values())

def get_degrees(G, weight=None):
    return list(dict(G.degree(weight=weight)).values())

def get_weight_distribution(graph):
    values = get_weights(graph)
    counts = Counter(values) 
    dist = dict(zip(counts.keys(), np.array(list(counts.values()))/len(graph.edges())))
    return dist, values

def get_degree_distribution(G):
    values = get_degrees(G)
    counts = Counter(values) 
    dist = dict(zip(counts.keys(), np.array(list(counts.values()))/len(G)))
    return dist, values

properties = {  'Weight':[get_weight_distribution,get_weights],
                'Degree':[get_degree_distribution, get_degrees]}


# %% 

for prop in properties:
    dist_res = pd.DataFrame(index=set_labels)

    # Loop through the Networks
    for csv_file in csv_files:
        try:
            g = originals[csv_file.split('.')[0]]        
            
            if len(g.edges()) >= 1000:

                
                vals = []
                for i, backbone in enumerate(set_labels):

                    backbone_g = backbones[csv_file.split('.')[0]][i]

                    if len(backbone_g.edges()) != len(g.edges()):


                        v = round(ks_2samp(properties[prop][1](g), properties[prop][1](backbone_g))[0],2)
                        vals.append(v)
                
                    else:
                        vals.append(None)



                dist_res[csv_file.split('.')[0]] = vals
               
        except:
                dist_res[csv_file.split('.')[0]] = [np.nan]*8

    os.makedirs('Results/Distribution/', exist_ok=True)
    with open('Results/Distribution/' + prop + '.pickle', 'wb') as f:
        pickle.dump(dist_res, f, pickle.HIGHEST_PROTOCOL)
    # break
# %%
# orders = {  'Weight':['DS', 'PMFG', 'MB', 'UMB', 'PLAM', 'MSP', 'HB'],
#             'Degree':['DS', 'MB', 'PMFG', 'UMB', 'HB', 'MSP', 'PLAM']}
# for prop in properties.keys():
#     colors = {'original':'#000000', 'MSP':'#4c72b0', 'DS':'#dd8452', 'PLAM':'#55a868', 'PMFG':'#c44e52', 'MB':'#8172b3', 'UMB':'#937860', 'HB':'#da8bc3', 'HSS':'#8c8c8c'}#, '#ccb974', '#64b5cd'}backbones = ['mlf_score', 'df_alpha', 'lans_alpha', 'gloss_alpha', 'ecm_alpha', 'pf_alpha', 'nc_alpha']

#     sns.set(font_scale = 0.5)

#     with open('Results/Distribution/' + prop + '.pickle', 'rb') as f:
#         dist_res = pickle.load(f)
#     res = dist_res.rank(method='dense')
#     res = res.T
#     res= res[orders[prop]]
#     plt.figure(figsize=(5,2.5))

#     sns.set_theme(style="darkgrid")
#     sns.set_palette([colors[name] for name in res.columns])


#     g = sns.boxplot(data=res, meanline=True, flierprops={"marker": "x"})

#     # g.set_xticklabels([set_labels[label] for label in res.columns])#, rotation=90)
#     g.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])


#     plt.tight_layout()
    # plt.savefig('/mnt/c/Users/Yassin/github/aliplosone/Figures/Weight Distribution/Weight-dist-ks-boxplot.png', dpi=300)
    # plt.savefig('/mnt/c/Users/Yassin/Desktop/Code/Backbones Survey 2022/Figures/Weight Distribution/Weight-dist-ks-boxplot-corrected.eps', dpi=300)
# %%


orders = {  'Weight':['DS', 'PMFG', 'MB', 'UMB', 'PLAM', 'MSP', 'HB'],
            'Degree':['DS', 'MB', 'PMFG', 'UMB', 'HB', 'MSP', 'PLAM']}
properties = ['Weight', 'Degree']  # replace with your actual properties

colors = {'original':'#000000', 'MSP':'#4c72b0', 'DS':'#dd8452', 'PLAM':'#55a868', 'PMFG':'#c44e52', 'MB':'#8172b3', 'UMB':'#937860', 'HB':'#da8bc3', 'HSS':'#8c8c8c'}

sns.set(font_scale = 1.2)  # Increase font scale

for prop in properties:
    with open('Results/Distribution/' + prop + '.pickle', 'rb') as f:
        dist_res = pickle.load(f)
    res = dist_res.rank(method='dense')
    res = res.T
    res= res[orders[prop]]
    plt.figure(figsize=(10,5))  # Increase figure size
    sns.set(font_scale = 1.4)  # Increase font scale

    sns.set_style("whitegrid")  # Use whitegrid style for better visibility
    sns.set_palette([colors[name] for name in res.columns])
    g = sns.boxplot(data=res, meanline=True, flierprops={"marker": "x"})  # Use box plot

    g.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
    # plt.title(f'Distribution of {prop}')  # Add title
    plt.xlabel('Method')  # Add x-label
    plt.ylabel('Rank')  # Add y-label

    plt.tight_layout()
    # plt.savefig('Figures/' + prop + '-distribution.png', dpi=300)
    plt.show()
# %%
for prop in properties:
    with open('Results/Distribution/' + prop + '.pickle', 'rb') as f:
        dist_res = pickle.load(f)
    res = dist_res.rank(method='dense')
    res = res.T
    res= res[orders[prop]]
    plt.figure(figsize=(10,5))  # Increase figure size
    sns.set(font_scale = 1.4)  # Increase font scale

    sns.set_style("whitegrid")  # Use whitegrid style for better visibility
    sns.set_palette([colors[name] for name in res.columns])
    g = sns.violinplot(data=res, inner="points", bw=.2, cut=0)  # Use violin plot

    g.set_yticks([1, 2, 3, 4, 5, 6, 7, 8])
    # plt.title(f'Distribution of {prop}')  # Add title
    plt.xlabel('Method')  # Add x-label
    plt.ylabel('Rank')  # Add y-label

    plt.tight_layout()
    # plt.savefig('Figures/' + prop + '-distribution.png', dpi=300)
    plt.show()
# %%
for prop in properties:
    with open('Results/Distribution/' + prop + '.pickle', 'rb') as f:
        dist_res = pickle.load(f)
    res = dist_res.T
    res= res[orders[prop]]
    plt.figure(figsize=(10,5))  # Increase figure size
    sns.set(font_scale = 1.4)  # Increase font scale

    sns.set_style("whitegrid")  # Use whitegrid style for better visibility
    sns.set_palette([colors[name] for name in res.columns])
    g = sns.violinplot(data=res, inner="points", bw=.2, cut=0)  # Use violin plot

    # plt.title(f'Distribution of {prop}')  # Add title
    plt.xlabel('Method')  # Add x-label
    plt.ylabel('KS Statistic')  # Add y-label

    plt.tight_layout()
    plt.savefig('Figures/' + prop + '-distribution-values.png', dpi=300)
    plt.show()

# %%
