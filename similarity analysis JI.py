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
def compute_jaccard(table1, table2):
    set1 = set(table1.edges)
    set2 = set(table2.edges)
    return float(len(set1.intersection(set2))) / len(set1.union(set2))

# %%
corr_dict = {}
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

        corr_df = pd.DataFrame(columns=set_labels, index=set_labels)
        for i, backbone1 in enumerate(backbones):
                for j, backbone2 in enumerate(backbones):
                        if j>i:
                                corr_df[set_labels[i]][set_labels[j]] = round(np.float(compute_jaccard(backbone1, backbone2)), 2)
                
        corr_dict[csv_file.split('.')[0]] = corr_df
    except:
        print(csv_file)
        continue


# %%
df_concat = pd.concat(corr_dict.values())

by_row_index = df_concat.groupby(df_concat.index)

df_means = by_row_index.mean()
df_means = df_means.reindex(df_means.columns)

df_variance = by_row_index.std()
df_variance = df_variance.reindex(df_variance.columns)

# Remove the first row and the last column
df_means = df_means.iloc[1:, :-1]
df_variance = df_variance.iloc[1:, :-1]

# %%
sns.set_theme(style="white", font_scale = 1.5)

fig, ax = plt.subplots(1, 2, figsize=(20,10))  # Increase figure size

# Use a diverging color palette
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
p = sns.heatmap(df_means, mask=df_means.isna(), cmap=cmap, vmax=1, vmin=0,
            square=True, linewidths=0, cbar_kws={"shrink": .5}, ax = ax[0], annot=True, annot_kws={"size":15})

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(df_variance, mask=df_variance.isna(), cmap=cmap, vmin=0,
            square=True, linewidths=0, cbar_kws={"shrink": .5}, ax = ax[1], annot=True, annot_kws={"size":15})

g.set_xticklabels(g.get_xticklabels(), rotation=0) 
g.set_yticklabels(g.get_yticklabels(), rotation=0) 
p.set_yticklabels(p.get_yticklabels(), rotation=0)


ax[0].set_title('Mean', weight='bold', fontsize=20)
ax[1].set_title('Standard Deviation', weight='bold', fontsize=20)

fig.subplots_adjust(wspace=3)
plt.tight_layout()

# plt.savefig('Figures/similarity-heatmap.png', dpi=300)

# plt.close()
    

# %%

# %%
correlations = {}

for file, matrix in corr_dict.items():
    methods = matrix.columns
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:  # This ensures we only get each pair once
                column_name = f"{method1}_{method2}"
                correlation = matrix[method1][method2]
                if column_name in correlations:
                    correlations[column_name].append(correlation)
                else:
                    correlations[column_name] = [correlation]

df = pd.DataFrame(correlations)

# %%
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming 'df' is your DataFrame

# cmap = plt.get_cmap('coolwarm')

# fig, axs = plt.subplots(4, 7, figsize=(15, 10))

# for i, ax in enumerate(axs.flatten()):
#     if i < len(df.columns):
#         column = df.columns[i]
#         mean = df[column].mean()
#         color = cmap(mean)  # Map the mean to a color
#         sns.violinplot(y=column, data=df, ax=ax, color=color)
#         ax.set_title(column)

#     else:
#         ax.set_visible(False)  # Hide unused subplots
#     ax.set_ylim(0, 1)  # Set the y-axis limits to [0, 1]

# plt.tight_layout()
# plt.show()

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
        # ax.axhline(mean, color='r', linestyle='--') 
        ax.plot(mean, 'ro') 
        

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
# plt.savefig('Figures/similarity-plot.png', dpi=300)
plt.show()

# %%
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming 'df' is your DataFrame

# cmap = plt.get_cmap('coolwarm')

# fig, axs = plt.subplots(7, 7, figsize=(15, 15))  # Increase the height of the figure
# counter = 0

# for row in range(7):
#     for col in range(7):
#         if col <= row and counter < len(df.columns):  # Only plot if col <= row
#             column = df.columns[counter]
#             mean = df[column].mean()
#             color = cmap(mean)  # Map the mean to a color
#             sns.violinplot(y=column, data=df, ax=axs[row, col], color=color)
#             axs[row, col].set_title(column)
#             axs[row, col].set_ylim(0, 1)  # Set the y-axis limits to [0, 1]
#             counter += 1
#         else:
#             axs[row, col].axis('off')  # Hide the unused subplots

# plt.tight_layout()
# plt.show()
# %%
# %%


# %%
