#!/usr/bin/env python
# coding: utf-8

# In[37]:


import torch
from torch_geometric.data import Data
import numpy as np
import scanpy as sc
import pandas as pd
from graph_construction import construct_graph
from tcn_autoencoder import train_and_extract_features
from data_splitting import create_train_val_tf, generate_train_val_kfold
from transformer_model import train_model, test_model
from test_split import split_edges, split_tfs


# In[38]:


file_name = "mESC"
adata = sc.read("/home/zfd297/benchmark/dataset/" +file_name + ".h5ad")

filtered_refnet = adata.uns["grn"]
fea_df = pd.DataFrame(adata.uns["gpt_emb"])
fea_df["Gene"] = adata.var_names
X_norm = adata.X.T
tfs = np.unique(filtered_refnet["Gene1"])


# In[39]:


data_orig, tfs_index, le = construct_graph(filtered_refnet, adata.var_names,fea_df, tfs)
data_orig


# In[40]:


tcauto_model, features = train_and_extract_features(X_norm,learning_rate=0.001, weight_decay=1e-4)
print(features.shape)


# In[41]:


reduced_features_df = pd.DataFrame(features.cpu().numpy())
reduced_features_df["Gene"] = adata.var_names
reduced_features_df

# create features of nodes
node_features_1 = fea_df.set_index('Gene').reindex(le.classes_).fillna(0).values
node_features_2 = reduced_features_df.set_index('Gene').reindex(le.classes_).fillna(0).values


x = torch.tensor(node_features_1, dtype=torch.float)
x_additional = torch.tensor(node_features_2, dtype=torch.float)
# build graph object
data = Data(x=x, x_additional=x_additional, edge_index=data_orig.edge_index)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
data


# In[42]:


train_val_sets_kfold = generate_train_val_kfold(data,10)


# In[43]:


train_val_sets_kfold


# In[44]:


train_val_sets = train_val_sets_kfold

results = []

for fold, (train_data, val_data) in enumerate(train_val_sets):
    print(f'Fold {fold + 1}:')
    
    # training
    model = train_model(
        train_data, 
        
        hidden_channels=64, 
        num_heads=16, 
        dropout=0.5, 
        lr=0.000005, 
        weight_decay=1e-3, 
        num_epochs=200, 
        print_interval=10
    )
    
    results.append({
        'fold': fold + 1,
       
        'model': model
    })
    
    print('--------------')



model_list = [r['model'] for r in results]
test_auc = [test_model(i, val_data) for i in model_list]
test_auc

