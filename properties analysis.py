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
    except:
        print(csv_file)
        continue



# %%
from netbone.measures import node_fraction, edge_fraction, number_connected_components, transitivity, average_clustering_coefficient
from scipy.stats import entropy
def reachability(original, G):
    if len(G)==0:
        return 0
    r = 0
    for c in [len(component) for component in nx.connected_components(G)]:
        r += c*(c-1)
    return r/(len(G)*(len(G) - 1))

def weights(G):
    return list(nx.get_edge_attributes(G, 'weight').values())
    
def weight_fraction(network, b):
    if len(b)==0:
        return 0
    return sum(weights(b))/sum(weights(network))

def weight_entropy(original, graph):
    return entropy(weights(graph), base=2)/entropy(weights(original), base=2)

def density(original, graph):
    backbone_density = (2*len(graph.edges()))/(len(original)*(len(original)-1))
    graph_density = (2*len(original.edges()))/(len(original)*(len(original)-1))
    return backbone_density/graph_density

def transitivity(original, graph):
    if nx.transitivity(original) == 0:
        return 0
    return nx.transitivity(graph)/nx.transitivity(original)
    
def degrees(G, weight=None):
    return list(dict(G.degree(weight=weight)).values())

# %%
properties = {  'Transitivity':transitivity, 
                'Density': density, 
                'Number of Components':number_connected_components, 
                'Edge Fraction':edge_fraction,
                'Weight Entropy':weight_entropy,  
                'Weight Fraction':weight_fraction, 
                'Node Fraction':node_fraction,  
                'Reachability':reachability}





# %%
def process_property_and_file(prop, csv_file):
    try:
        # values = [properties[prop](nx.read_weighted_edgelist(os.path.join('Data', csv_file + '.csv')), backbone) for backbone in backbones[csv_file] if len(backbone) > 0 else np.nan] 
        values = [properties[prop](nx.read_weighted_edgelist(os.path.join('Data', csv_file + '.csv')), backbone) if len(backbone) > 0 else np.nan for backbone in backbones[csv_file]]
        return [prop, csv_file, values]
    except Exception as e:
        values =  [np.nan for backbone in backbones[csv_file]]
        return [prop, csv_file, values]
        print(f"An error occurred in file: {e}")
    

# %% 
prop_dict = {prop: pd.DataFrame(columns=[csv_file.split('.')[0] for csv_file in csv_files], index=set_labels) for prop in properties.keys()}

def process_property(prop):
    try:
        with ThreadPoolExecutor(max_workers=39) as executor:
            futures = {executor.submit(process_property_and_file, prop, csv_file) for csv_file in backbones.keys()}
            results = [f.result() for f in tqdm(as_completed(futures), total=len(futures))]
    except Exception as e:
        print(f"An error occurred in prop: {type(e)}, {e}")
        return

    return results

with ThreadPoolExecutor(8) as executor:
    try:
        futures = {executor.submit(process_property, prop) for prop in properties.keys()}
        # results = [f.result() for f in tqdm(as_completed(futures), total=len(futures))]
    except Exception as e:
        print(f"An error occurred: {e}")

    for f in tqdm(as_completed(futures), total=len(futures)):
        for res in f.result():
            # print(res[0], res[1], res[2])
            prop_dict[res[0]][res[1]] = res[2]

        os.makedirs('Results/Properties/', exist_ok=True)
        with open('Results/Properties/' + res[0] + '.pickle', 'wb') as f:
            pickle.dump(prop_dict[res[0]], f, pickle.HIGHEST_PROTOCOL)
            
# %%
original = [len(nx.read_weighted_edgelist(os.path.join('Data', csv_file))) for csv_file in csv_files]


# %%
# colors = {'original':'#000000', 'MSP':'#4c72b0', 'DS':'#dd8452', 'PLAM':'#55a868', 'PMFG':'#c44e52', 'MB':'#8172b3', 'UMB':'#937860', 'HB':'#da8bc3', 'HSS':'#8c8c8c'}#, '#ccb974', '#64b5cd'}backbones = ['mlf_score', 'df_alpha', 'lans_alpha', 'gloss_alpha', 'ecm_alpha', 'pf_alpha', 'nc_alpha']

# for prop in properties.keys():
#     with open('Results/Properties/' + prop + '.pickle', 'rb') as f:
#         res = pickle.load(f).T
#     res['Nodes'] = original
#     res = res.sort_values(by='Nodes')

#     # res.fillna(-.5, inplace=True)
#     # res = res.replace(0, np.nan)
#     # fig, axes = plt.subplots(2,4, figsize=(12, 6), sharey=True)
#     fig = plt.figure(figsize=(12, 6))#, sharey=True)
  
#     # for backbone, subplot in zip(backbones, axes.flatten()):

#     subplots = []
#     for i, backbone in enumerate(set_labels):
#         subplot = fig.add_subplot(2, 4, i+1)

#         subplot.scatter(res.index, res[backbone], color=colors[backbone]) 
#         subplot.tick_params(labelbottom=False)
#         # subplot.set_title(set_labels[i])# + ' Rate= ' + str(round(res[res[backbone].notna()][backbone].count()/res.shape[0], 2)))
#         subplots.append(subplot)
        
#     y_min = min(subplot.get_ylim()[0] for subplot in subplots)
#     y_max = max(subplot.get_ylim()[1] for subplot in subplots)

#     for subplot in subplots:
#         subplot.set_ylim(y_min, y_max)
#         # subplot.set_yscale('log')
#         # subplot.legend()

#     fig.suptitle(prop)
#     # if file == 'Number of Components.pkl':
#     # plt.yscale('log')
#     fig.tight_layout() 
#     # fig.savefig('/mnt/c/Users/Yassin/github/aliplosone/variables/uncorrected/' + file.split('.')[0], dpi=300)
#     # break
# %%

# Set seaborn style and increase font size
sns.set(style="whitegrid", font_scale=1)
colors = {'original':'#000000', 'MSP':'#4c72b0', 'DS':'#dd8452', 'PLAM':'#55a868', 'PMFG':'#c44e52', 'MB':'#8172b3', 'UMB':'#937860', 'HB':'#da8bc3', 'HSS':'#8c8c8c'}#, '#ccb974', '#64b5cd'}backbones = ['mlf_score', 'df_alpha', 'lans_alpha', 'gloss_alpha', 'ecm_alpha', 'pf_alpha', 'nc_alpha']

for prop in properties.keys():
    with open('Results/Properties/' + prop + '.pickle', 'rb') as f:
        res = pickle.load(f).T
    res['Nodes'] = original
    res = res.sort_values(by='Nodes')

    fig = plt.figure(figsize=(10, 5))

    subplots = []
    for i, backbone in enumerate(set_labels):
        subplot = fig.add_subplot(2, 4, i+1)

        subplot.scatter(res.index, res[backbone], color=colors[backbone]) 
        subplot.tick_params(labelbottom=False)
        subplot.set_title(set_labels[i])
        subplot.grid(False)  # Add gridlines
        subplots.append(subplot)
        
    y_min = min(subplot.get_ylim()[0] for subplot in subplots)
    y_max = max(subplot.get_ylim()[1] for subplot in subplots)

    for subplot in subplots:
        subplot.set_ylim(y_min, y_max)

    fig.suptitle(prop)
    fig.tight_layout() 
    # plt.savefig('Figures/' + prop + '.png', dpi=300)
    # break
# %%
    

sns.set(style="whitegrid", font_scale=1.4)
colors = {'original':'#000000', 'MSP':'#4c72b0', 'DS':'#dd8452', 'PLAM':'#55a868', 'PMFG':'#c44e52', 'MB':'#8172b3', 'UMB':'#937860', 'HB':'#da8bc3', 'HSS':'#8c8c8c'}

for i, prop in enumerate(properties.keys()):
    with open('Results/Properties/' + prop + '.pickle', 'rb') as f:
        res = pickle.load(f).T
    res['Nodes'] = original
    res = res.melt(id_vars='Nodes', var_name='Backbone', value_name='Value')
    res = res.astype({'Value': float})
    # values = res['Nodes'].values
    # normalized_sizes = np.interp(values, (values.min(), values.max()), (1, 100))
    # res['Sizes'] = normalized_sizes
    plt.figure(figsize=(10, 5))
    sns.violinplot(x='Backbone', y='Value', 
                   data=res, 
                   palette=[colors[backbone] for backbone in res.Backbone.unique()], 
                   inner="points", 
                   bw=.2, 
                   cut=0)
    # sns.violinplot(x='Backbone', y='Value', 
    #                data=res, 
    #                palette=[colors[backbone] for backbone in res.Backbone.unique()],
    #                inner='None',
    #                bw=.2, 
    #                cut=0,)
    # plt.scatter(x='Backbone', y='Value', data=res, s='Sizes', color='black', alpha=0.7)
    # plt.title(prop)
    plt.xlabel('Method')
    plt.ylabel(prop)
    plt.yticks(np.arange(0, res['Value'].max() + 0.1, 0.1))  # Set y-ticks

    # plt.grid(False)  # Add gridlines
    plt.tight_layout()
    # plt.savefig('Figures/' + prop + '.png', dpi=300)
    plt.show()
    # break
# %%
for i, prop in enumerate(properties.keys()):
    with open('Results/Properties/' + prop + '.pickle', 'rb') as f:
        res = pickle.load(f).T

    # drop the rows that are all na values in res
    res = res.dropna(axis=0, how='all')
    res['Nodes'] = original
    res = res.melt(id_vars='Nodes', var_name='Backbone', value_name='Value')
    res = res.astype({'Value': float})

    # Calculate mean values and sort methods by these values
    order = res.groupby('Backbone')['Value'].mean().sort_values().index.tolist()

    plt.figure(figsize=(10, 5))
    sns.violinplot(x='Backbone', y='Value', 
                   data=res, 
                   palette=[colors[backbone] for backbone in order], 
                   inner="points", 
                   bw=.2, 
                   cut=0,
                   order
                   =order)  # Use the sorted list as the order
    sns.pointplot(x='Backbone', y='Value', 
                  data=res, 
                  order=order, 
                  color='red', 
                  scale=0.5, 
                  ci=None,
                  join=False)  # Plot the average of each method as a point
    plt.xlabel('Method')
    plt.ylabel(prop)
    if prop == 'Number of Components':
        plt.yscale('log')  
        plt.ylabel(f'{prop} in a log scale')
        plt.tight_layout()
        plt.savefig('Figures/' + prop + '.png', dpi=300)
        plt.show()
    # plt.yticks(np.arange(0, res['Value'].max()+0.01, 0.2))  # Set y-ticks
    plt.tight_layout()
    # plt.savefig('Figures/' + prop + '-2.png', dpi=300)
    plt.show()
# %%

# %%
