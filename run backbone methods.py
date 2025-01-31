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
import networkx as nx
from netbone.filters import boolean_filter
from netbone.backbone import Backbone
from pandas import DataFrame
from networkx import Graph
from netbone.utils.utils import edge_properties

# %%
# Get a list of all CSV files in the data directory
csv_files = [f for f in os.listdir('Data') if f.endswith('.csv')]

methods_dict = {nb.maximum_spanning_tree:"MSP", nb.plam:"PLAM", nb.pmfg:"PMFG", nb.doubly_stochastic:"Doubly Stochastic", nb.h_backbone:"H-Backbone", nb.metric_distance_backbone:"Metric Distance Backbone", nb.ultrametric_distance_backbone:"Ultrametric Distance Backbone", nb.high_salience_skeleton:"High Salience Skeleton"}

# %%


def run_method(csv_file, method):

    network = nx.read_weighted_edgelist(os.path.join('Data', csv_file))
    backbone = method(network)

    # Let's assume `dir_name` is the directory you want to create
    dir_name = 'Results/' + methods_dict[method]
    os.makedirs(dir_name, exist_ok=True)

    with open('Results/' + methods_dict[method] + "/" + csv_file.split('.')[0] + '.pickle', 'wb') as f:
        pickle.dump(backbone, f, pickle.HIGHEST_PROTOCOL)


# %%    

for method in methods_dict.keys():
    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=39) as executor:
        # Use the executor to submit tasks and get a list of futures
        futures = {executor.submit(run_method, csv_file, method) for csv_file in csv_files}

        # Use tqdm with as_completed to update the progress bar as tasks complete
        for f in tqdm(as_completed(futures), total=len(futures)):
            pass  # Or do something with f.result()
# %%
