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
# def overlap_coefficient(tabel_1, Tabel_2):
#     set_A = set(tabel_1.edges)
#     set_B = set(Tabel_2.edges)
#     intersection_size = len(set_A.intersection(set_B))
#     oc_score = intersection_size / min(len(set_A), len(set_B))
#     if len(set_A) < len(set_B):
#         return oc_score, 0
#     if len(set_A) > len(set_B):
#         return oc_score, 1
#     if len(set_A) == len(set_B):
#         return oc_score, 2

def overlap_coefficient(tabel_1, Tabel_2):
    set_A = set(tabel_1.edges)
    set_B = set(Tabel_2.edges)
    if len(set_A) == 0 or len(set_B) == 0:
        return 0
    intersection_size = len(set_A.intersection(set_B))
    oc_score = intersection_size / len(set_A)
    return oc_score
    
# %%
oc_dict = {}

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

        backbones = [msp_df, plam_df, pmfg_df, mb_df, umb_df, hb_df, hss_df, ds_df]

        oc_df = pd.DataFrame(columns=set_labels, index=set_labels)
        for i, backbone1 in enumerate(backbones):
            for j, backbone2 in enumerate(backbones):
                oc_df[set_labels[i]][set_labels[j]] = overlap_coefficient(backbone1, backbone2)
        oc_dict[csv_file.split('.')[0]] = oc_df
    except Exception as e:
        print(csv_file, e)
        continue

# %%
    
# Calculate the mean and standard deviation
df_means = pd.DataFrame(0, columns=set_labels, index=set_labels)
df_squares = pd.DataFrame(0, columns=set_labels, index=set_labels)

for oc_df in oc_dict.values():
    df_means += oc_df
    df_squares += oc_df**2

df_means /= len(oc_dict)
df_squares /= len(oc_dict)

df_stds = (df_squares - df_means**2)**0.5

df_means = df_means.astype(float)
df_stds = df_stds.astype(float)


# Set the font scale
# sns.set(font_scale=1.2)  # Increase the font scale to 1.5

# Create a mask for the diagonal
mask = np.eye(df_means.shape[0], dtype=bool)

# Create annotation arrays
annot_means = df_means.where(~mask).round(2).astype(str).replace('nan', 'X')
annot_stds = df_stds.where(~mask).round(2).astype(str).replace('nan', 'X')

# Plot the heatmaps
fig, ax = plt.subplots(1, 2, figsize=(11.5,5))

# Plot the heatmaps with the 'coolwarm' colormap
sns.heatmap(df_means, ax=ax[0], annot=annot_means.values, fmt="", cmap='coolwarm')
sns.heatmap(df_stds, ax=ax[1], annot=annot_stds.values, fmt="", cmap='coolwarm')

# Plot the diagonal cells with the 'grey' colormap
sns.heatmap(df_means, ax=ax[0], mask=~mask, annot=annot_means.values, fmt="", cmap=[(0.5, 0.5, 0.5)], cbar=False)
sns.heatmap(df_stds, ax=ax[1], mask=~mask, annot=annot_stds.values, fmt="", cmap=[(0.5, 0.5, 0.5)], cbar=False)

# Add text under each number in the heatmaps
# for i in range(df_means.shape[0]):
#     for j in range(df_means.shape[1]):
#         if i != j:
#             ax[0].text(j+0.5, i+0.8, df_means.columns[j] + ' < ' + df_means.index[i], ha='center', va='top')
#             ax[1].text(j+0.5, i+0.8, df_stds.columns[j] + ' < ' + df_stds.index[i], ha='center', va='top')

ax[0].set_title('Mean Overlap Coefficient')
ax[1].set_title('Standard Deviation of Overlap Coefficient')

plt.tight_layout()
plt.savefig('Figures/overlap-coefficient-1.png', dpi=300)
plt.show()
# # Create a mask for the diagonal
# mask = np.eye(df_means.shape[0], dtype=bool)

# # Create annotation arrays
# annot_means = df_means.where(~mask).round(2).astype(str).replace('nan', 'X')
# annot_stds = df_stds.where(~mask).round(2).astype(str).replace('nan', 'X')

# # Plot the heatmaps
# fig, ax = plt.subplots(1, 2, figsize=(15,5))

# # Plot the heatmaps with the 'coolwarm' colormap
# sns.heatmap(df_means, ax=ax[0], annot=annot_means.values, fmt="", cmap='coolwarm')
# sns.heatmap(df_stds, ax=ax[1], annot=annot_stds.values, fmt="", cmap='coolwarm')

# # Plot the diagonal cells with the 'grey' colormap
# sns.heatmap(df_means, ax=ax[0], mask=~mask, annot=annot_means.values, fmt="", cmap=[(0.5, 0.5, 0.5)], cbar=False)
# sns.heatmap(df_stds, ax=ax[1], mask=~mask, annot=annot_stds.values, fmt="", cmap=[(0.5, 0.5, 0.5)], cbar=False)

# ax[0].set_title('Mean Overlap Coefficient')
# ax[1].set_title('Standard Deviation of Overlap Coefficient')

# plt.savefig('Figures/overlap-coefficient-1.png', dpi=300)
# plt.show()









# %%
import itertools

# Create pairs
# pairs = list(itertools.permutations(set_labels, 2))
pairs = list(itertools.combinations(set_labels, 2))

# Convert pairs to strings
pairs = [f'{pair[0]}-{pair[1]}' for pair in pairs]

# %%
correlations = {}

for file, matrix in corr_dict.items():
    for pair in pairs:
        method1, method2 = pair.split('-')
        column_name = f"{method1}-{method2}"
    

        correlation = matrix[method1][method2]

        if column_name in correlations:
            correlations[column_name].append(correlation)
        else:
            correlations[column_name] = [correlation]
 
df = pd.DataFrame(correlations)

# %%
ids = {}

for file, matrix in oc_b_dict.items():
    for pair in pairs:
        method1, method2 = pair.split('-')
        column_name = f"{method1}-{method2}"
    

        correlation = matrix[method1][method2]

        if column_name in ids:
            ids[column_name].append(correlation)
        else:
            ids[column_name] = [correlation]
 
dfs = pd.DataFrame(ids)

# %%
# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame

cmap = plt.get_cmap('coolwarm')

fig, axs = plt.subplots(4, 7, figsize=(15, 10))

for i, ax in enumerate(axs.flatten()):
    if i < len(dfs.columns):
        column = dfs.columns[i]
        mean = dfs[column].mean()
        color = cmap(mean)  # Map the mean to a color
        sns.violinplot(y=column, data=dfs, ax=ax, color=color)
        ax.set_title(column)

    else:
        ax.set_visible(False)  # Hide unused subplots
    ax.set_ylim(0, 1)  # Set the y-axis limits to [0, 1]

plt.tight_layout()
plt.show()

# %%

cmap = plt.get_cmap('coolwarm')

# Calculate the mean of each column and sort the columns by their mean in descending order
means = df.mean()
sorted_columns = means.sort_values(ascending=False).index

fig, axs = plt.subplots(4, 7, figsize=(15, 10))

for i, ax in enumerate(axs.flatten()):
    if i < len(sorted_columns):
        column = sorted_columns[i]
        mean = means[column]
        color = cmap(mean)  # Map the mean to a color
        sns.violinplot(y=column, data=df, ax=ax, color=color, inner="points", bw =.2, cut=0)
        ax.set(ylabel=None)
        ax.set_title(column)
        ax.axhline(mean, color='r', linestyle='--') 
        

    else:
        ax.set_visible(False)  # Hide unused subplots
    ax.set_ylim(0, 1)  # Set the y-axis limits to [0, 1]
    
    ax.tick_params(top=False,
               bottom=False,
               left=True,
               right=False,
               labelleft=True,
               labelbottom=False)
    
plt.tight_layout(pad=1)

# Add a color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.05)


os.makedirs('Figures', exist_ok=True)
# plt.savefig('Figures/similarity-plot-B.png', dpi=300)
plt.show()

# %%
import itertools

cmap = plt.get_cmap('coolwarm')

# Create pairs
pairs = list(itertools.combinations(set_labels, 2))

fig, axs = plt.subplots(4,7, figsize=(20, 10))

for i, ax in enumerate(axs.flatten()):
    pair_keys = pairs[i]
    pair =  list(itertools.permutations([i for i in pair_keys], 2))
    pair = [f'{p[0]}-{p[1]}' for p in pair]

    pair_df = df[list(pair)].melt(var_name='column')
    mean1, mean2 = df[pair[0]].mean(), df[pair[1]].mean()
    color = [cmap(mean1), cmap(mean2)]  # Map the mean to a color

    sns.violinplot(x='column', y='value', data=pair_df, ax=ax, inner="points", color=color, bw=.2, cut=0, palette=cmap([mean1, mean2]))
    ax.set_title(f'{pair[0]}')
    ax.scatter([0], [mean1], color='brown', s=100, label='mean1')
    ax.scatter([1], [mean2], color='b', s=100, label='mean2')
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    # for column in pair:
    #     mean = df[column].mean()
    #     color = cmap(mean)  # Map the mean to a color
    #     sns.violinplot(y=column, data=df, ax=ax, color=color, inner="points", bw=.2, cut=0)
    #     ax.set(ylabel=None)
    #     ax.set_title(f'{pair[0]}-{pair[1]}')
    #     ax.axhline(mean, color='r', linestyle='--')

    ax.set_ylim(0, 1)  # Set the y-axis limits to [0, 1]
    ax.tick_params(top=False, bottom=False, left=True, right=False, labelleft=True, labelbottom=False)

plt.tight_layout(pad=1)

# Add a color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.05)

os.makedirs('Figures', exist_ok=True)
plt.savefig('Figures/similarity-plot-SS.png', dpi=300)
plt.show()

# %%
