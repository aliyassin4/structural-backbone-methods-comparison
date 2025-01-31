# %% 
import os
import pickle
from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
import netbone as nb
from concurrent.futures import ThreadPoolExecutor, as_completed
import seaborn as sns
from scipy.stats import pointbiserialr
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

set_labels = ['MSP', 'PLAM', 'PMFG', "MB", 'UMB', 'HB', 'HSS', 'DS']
# %%
def edges(row):
    if str(int(row.source)) < str(int(row.target)):
        return str(int(row.source)) + '-' + str(int(row.target))
    return str(int(row.target)) + '-' + str(int(row.source))

for csv_file in csv_files:
    try:
        df = pd.read_csv(os.path.join('Results/computed props', csv_file))[['source', 'target', 'weight', 'betweenness', 'weighted_betweenness']]
        G = nx.from_pandas_edgelist(df, edge_attr='weight')
        degrees = G.degree()
        df['degree'] = df.apply(lambda row: degrees[row.source] * degrees[row.target], axis=1)
        
        degrees = G.degree(weight='weight')
        df['weighted-degree'] = df.apply(lambda row: degrees[row.source] * degrees[row.target], axis=1)

        with open('Results/Combined/' + csv_file.split('.')[0] + '.pickle', 'rb') as f:
            data = pickle.load(f)

        df['edges'] = df.apply(lambda row: edges(row), axis=1)
        df = df.drop(columns=['source', 'target','weight'])
        merged_df = df.merge(data, on='edges')

        os.makedirs('Results/Edge Properties/', exist_ok=True)
        merged_df.to_csv(os.path.join('Results/Edge Properties', csv_file), index=False)
    except Exception as e:
        print(csv_file, e)
        continue
    # break

# %%
def compute_correlation(column_name):
    corr_df = pd.DataFrame(index=set_labels)
    for csv_file in csv_files:
        try:
            data = pd.read_csv(os.path.join('Results/Edge Properties', csv_file))
            corr_df[csv_file.split('.')[0]] = [pointbiserialr(data['in_backbone_' + method], data[column_name])[0] for method in set_labels]

        except Exception as e:
            print(csv_file, e)
            
    return corr_df

# %%
properties = ['weight', 'degree', 'weighted-degree', 'betweenness', 'weighted_betweenness']
for prop in properties:
    corr_df = compute_correlation(prop)
    os.makedirs('Results/Properties Correlation/', exist_ok=True)
    with open('Results/Properties Correlation/' + prop + '.pickle', 'wb') as f:
        pickle.dump(corr_df.abs(), f)
    # break


# %%
# for prop in properties:
#     colors = {'original':'#000000', 'MSP':'#4c72b0', 'DS':'#dd8452', 'PLAM':'#55a868', 'PMFG':'#c44e52', 'MB':'#8172b3', 'UMB':'#937860', 'HB':'#da8bc3', 'HSS':'#8c8c8c'}#, '#ccb974', '#64b5cd'}backbones = ['mlf_score', 'df_alpha', 'lans_alpha', 'gloss_alpha', 'ecm_alpha', 'pf_alpha', 'nc_alpha']
#     plt.figure(figsize=(5,2.5))
#     sns.set_theme(style="darkgrid")
#     with open('Results/Properties Correlation/' + prop + '.pickle', 'rb') as f:
#         corr_df = pickle.load(f)

#     sns.set_palette([colors[name] for name in corr_df.T.columns])


#     g = sns.boxplot(data=corr_df.T, meanline=True, flierprops={"marker": "x"})

#     # g.set_xticklabels([set_labels[label] for label in corr_df.T.columns])#, rotation=90)
#     g.set_yticks(np.arange(0, 1.2, step=0.2))
#     # b.set_yticklabels(b.get_yticks(), size = 15)
    
#     plt.title(f'{prop.capitalize()}')
    
#     plt.tight_layout()
#     os.makedirs('Figures', exist_ok=True)
#     plt.savefig(f'Figures/{prop}-correlation.png', dpi=300)

# %%
# properties_labels = ['Weight', 'Degree', 'Weighted Degree', 'Betweenness', 'Weighted Betweenness']
        
# colors = {'original':'#000000', 'MSP':'#4c72b0', 'DS':'#dd8452', 'PLAM':'#55a868', 'PMFG':'#c44e52', 'MB':'#8172b3', 'UMB':'#937860', 'HB':'#da8bc3', 'HSS':'#8c8c8c'}

# sns.set_theme(style="darkgrid")

# fig, axs = plt.subplots(3, 2, figsize=(9, 8))

# for i, prop in enumerate(properties):
#     with open('Results/Properties Correlation/' + prop + '.pickle', 'rb') as f:
#         corr_df = pickle.load(f)

#     sns.set_palette([colors[name] for name in corr_df.T.columns])

#     if i == 0:  # First plot centered in the first row
#         ax = plt.subplot2grid((3, 2), (0, 0), colspan=2)
#     else:  # Other plots in two rows with two columns
#         ax = plt.subplot2grid((3, 2), ((i+1)//2, (i+1)%2))

#     g = sns.boxplot(data=corr_df.T, meanline=True, flierprops={"marker": "x"})

#     g.set_yticks(np.arange(0, 1.2, step=0.2))
#     plt.title(f'{properties_labels[i].capitalize()}')

# plt.tight_layout()
# os.makedirs('Figures', exist_ok=True)
# # plt.savefig('Figures/properties-correlation.png', dpi=300)
# plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
properties = ['weight', 'degree', 'weighted-degree', 'betweenness', 'weighted_betweenness']

properties_labels = ['Weight', 'Degree', 'Weighted Degree', 'Betweenness', 'Weighted Betweenness']

colors = {'original':'#000000', 'MSP':'#4c72b0', 'DS':'#dd8452', 'PLAM':'#55a868', 'PMFG':'#c44e52', 'MB':'#8172b3', 'UMB':'#937860', 'HB':'#da8bc3', 'HSS':'#8c8c8c'}


fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # Increase figure size

for i, prop in enumerate(properties):
    with open('Results/Properties Correlation/' + prop + '.pickle', 'rb') as f:
        corr_df = pickle.load(f)
    sns.set(font_scale=1.4)  # Increase font scale

    sns.set_style("whitegrid")  # Use whitegrid style for better visibility

    sns.set_palette([colors[name] for name in corr_df.T.columns])

    if i == 0:  # First plot centered in the first row
        ax = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    else:  # Other plots in two rows with two columns
        ax = plt.subplot2grid((3, 2), ((i+1)//2, (i+1)%2))

    g = sns.boxplot(data=corr_df.T, meanline=True, flierprops={"marker": "x"})

    g.set_yticks(np.arange(0, 1.2, step=0.2))
    plt.title(f'{properties_labels[i].capitalize()}')  # Add title
    plt.xlabel('Method')  # Add x-label
    plt.ylabel('Correlation')  # Add y-label

plt.tight_layout()
os.makedirs('Figures', exist_ok=True)
# plt.savefig('Figures/properties-correlation.png', dpi=300)
plt.show()
# %%
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pickle
# import numpy as np

# # Assuming 'properties' is a list of your properties

# colors = {'original':'#000000', 'MSP':'#4c72b0', 'DS':'#dd8452', 'PLAM':'#55a868', 'PMFG':'#c44e52', 'MB':'#8172b3', 'UMB':'#937860', 'HB':'#da8bc3', 'HSS':'#8c8c8c'}

# sns.set_theme(style="darkgrid")

# for prop in properties:
#     plt.figure(figsize=(8,3))

#     with open('Results/Properties Correlation/' + prop + '.pickle', 'rb') as f:
#         corr_df = pickle.load(f)

#     sns.set_palette([colors[name] for name in corr_df.T.columns])

#     g = sns.violinplot(data=corr_df.T)

#     g.set_yticks(np.arange(0, 1.2, step=0.2))
#     plt.title(f'{prop.capitalize()}')

#     plt.tight_layout()
#     plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
properties = ['weight', 'degree', 'weighted-degree', 'betweenness', 'weighted_betweenness']

properties_labels = ['Weight', 'Degree', 'Weighted Degree', 'Betweenness', 'Weighted Betweenness']

colors = {'original':'#000000', 'MSP':'#4c72b0', 'DS':'#dd8452', 'PLAM':'#55a868', 'PMFG':'#c44e52', 'MB':'#8172b3', 'UMB':'#937860', 'HB':'#da8bc3', 'HSS':'#8c8c8c'}

for i, prop in enumerate(properties):
    with open('Results/Properties Correlation/' + prop + '.pickle', 'rb') as f:
        corr_df = pickle.load(f)
    sns.set(font_scale=1.4)  # Increase font scale
    
    sns.set_style("whitegrid")  # Use whitegrid style for better visibility
    plt.figure(figsize=(10, 5))

    g = sns.violinplot(data=corr_df.T, inner="points", palette=[colors[name] for name in corr_df.T.columns], bw=.2, cut=0)  # Use violin plot instead of box plot
    
    sns.pointplot(data=corr_df.T, color='red', scale=0.5, ci=None, join=False)

    g.set_yticks(np.arange(0, 1.2, step=0.2))
    # plt.title(f'{properties_labels[i]}', weight='bold')  # Add title
    # plt.xlabel('Method')  # Add x-label
    plt.ylabel('Correlation')  # Add y-label

    plt.tight_layout()

    os.makedirs('Figures', exist_ok=True)
    plt.savefig('Figures/'+prop+'.png', dpi=300)
    plt.show()



# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
properties = ['weight', 'degree', 'betweenness', 'weighted-degree', 'weighted_betweenness']

properties_labels = ['Weight', 'Degree', 'Betweenness', 'Weighted Degree', 'Weighted Betweenness']

colors = {'original':'#000000', 'MSP':'#4c72b0', 'DS':'#dd8452', 'PLAM':'#55a868', 'PMFG':'#c44e52', 'MB':'#8172b3', 'UMB':'#937860', 'HB':'#da8bc3', 'HSS':'#8c8c8c'}

fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # Increase figure size

for i, prop in enumerate(properties):
    with open('Results/Properties Correlation/' + prop + '.pickle', 'rb') as f:
        corr_df = pickle.load(f)
    sns.set(font_scale=1.4)  # Increase font scale

    sns.set_style("whitegrid")  # Use whitegrid style for better visibility

    if i == 0:  # First plot centered in the first row
        ax = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    else:  # Other plots in two rows with two columns
        ax = plt.subplot2grid((3, 2), ((i+1)//2, (i+1)%2))

    g = sns.violinplot(data=corr_df.T, inner="points", palette=[colors[name] for name in corr_df.T.columns], bw=.2, cut=0)  # Use violin plot instead of box plot
    
    sns.pointplot(data=corr_df.T, color='red', scale=0.5, ci=None, join=False)

    g.set_yticks(np.arange(0, 1.2, step=0.2))
    plt.title(f'{properties_labels[i]}', weight='bold')  # Add title
    # plt.xlabel('Method')  # Add x-label
    plt.ylabel('Correlation')  # Add y-label

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)  # Increase vertical spacing

os.makedirs('Figures', exist_ok=True)
plt.savefig('Figures/properties-correlation-2.png', dpi=300)
plt.show()
# %%



fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Create 2x2 grid of subplots

for i, prop in enumerate(properties[1:]):  # Skip the first property
    with open('Results/Properties Correlation/' + prop + '.pickle', 'rb') as f:
        corr_df = pickle.load(f)
    ax = plt.subplot2grid((2, 2), (i//2, i%2))  # Arrange the plots in a 2x2 grid

    g = sns.violinplot(data=corr_df.T, inner="points", palette=[colors[name] for name in corr_df.T.columns], bw=.2, cut=0, ax=ax)  # Use violin plot instead of box plot
    
    sns.pointplot(data=corr_df.T, color='red', scale=0.5, ci=None, join=False, ax=ax)

    g.set_yticks(np.arange(0, 1.2, step=0.2))
    ax.set_title(f'{properties_labels[i+1]}', weight='bold')  # Add title
    ax.set_ylabel('Correlation')  # Add y-label

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)  # Increase vertical spacing

os.makedirs('Figures', exist_ok=True)
plt.savefig('Figures/properties-correlation-combined.png', dpi=300)
plt.show()
# %%
