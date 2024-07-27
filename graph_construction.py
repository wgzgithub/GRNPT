# graph_construction.py

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

def construct_graph(filtered_refnet, genes, fea_df, tfs):
    np.random.seed(42)

    # 创建边索引
    le = LabelEncoder()
    #genes = pd.concat([filtered_refnet['Gene1'], filtered_refnet['Gene2']])
    le.fit(genes)
    edge_index = torch.tensor([le.transform(filtered_refnet['Gene1']), 
                               le.transform(filtered_refnet['Gene2'])], dtype=torch.long)

    # 创建节点特征
    node_features = fea_df.set_index('Gene').reindex(le.classes_).fillna(0).values
    x = torch.tensor(node_features, dtype=torch.float)

    # 构建图数据对象
    data = Data(x=x, edge_index=edge_index)

    # 移动到可用设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # 转换tfs
    tfs_index = le.transform(tfs)

    return data, tfs_index, le

# 如果需要直接运行此文件进行测试，可以添加以下代码
if __name__ == "__main__":
    # 这里应该添加测试代码，例如：
    # filtered_refnet = pd.read_csv('path_to_filtered_refnet.csv')
    # fea_df = pd.read_csv('path_to_fea_df.csv')
    # tfs = pd.read_csv('path_to_tfs.csv')['TF'].tolist()
    # 
    # data, tfs_index, le = construct_graph(filtered_refnet, fea_df, tfs)
    # print(data)
    # print(tfs_index)
    pass