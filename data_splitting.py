import torch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.utils import negative_sampling

def create_train_val_tf(data, tfs_index):
    if data.edge_index.is_cuda:
        data.edge_index = data.edge_index.cpu()

    edges = data.edge_index.t().numpy()
    num_edges = edges.shape[0]
    train_val_sets = []

    for tf_node in tfs_index:
        val_mask = np.isin(edges[:, 0], tf_node)
        val_edges = edges[val_mask]

        if len(val_edges) == 0:
            continue  # 如果没有边与该 tf_node 相连，跳过

        val_edge_index = torch.tensor(val_edges.T, dtype=torch.long)
        num_positive_samples = val_edge_index.size(1)
        num_neg_samples = num_positive_samples
        neg_samples = negative_sampling(edge_index=data.edge_index, num_nodes=data.x.size(0), num_neg_samples=num_neg_samples)
        
        edge_to_move = val_edge_index[:, 0].unsqueeze(1)
        remaining_val_edge_index = val_edge_index[:, 1:]

        val_edge_label_index = torch.cat([remaining_val_edge_index, neg_samples], dim=1)
        val_edge_label = torch.cat([torch.ones(remaining_val_edge_index.size(1)), torch.zeros(num_neg_samples)], dim=0)

        val_data = Data(
            x=data.x, 
            edge_label_index=val_edge_label_index, 
            edge_label=val_edge_label, 
            x_additional=data.x_additional
        )

        num_train_edges = max(1, int(0.1 * num_edges))

        remaining_edges = edges[~val_mask]
        selected_train_edges = remaining_edges[np.random.choice(remaining_edges.shape[0], num_train_edges - 1, replace=False)]

        train_edge_index = torch.tensor(selected_train_edges.T, dtype=torch.long)
        train_edge_index = torch.cat([train_edge_index, edge_to_move], dim=1)

        moved_edge_idx = train_edge_index.size(1) - 1
        num_edges_to_remove = max(1, int(0.5 * (train_edge_index.size(1) - 1)))
        remaining_indices = np.random.choice(train_edge_index.size(1) - 1, train_edge_index.size(1) - 1 - num_edges_to_remove, replace=False)
        remaining_indices = np.append(remaining_indices, moved_edge_idx)
        train_edge_index = train_edge_index[:, remaining_indices]

        train_edge_label_index = train_edge_index
        train_edge_label = torch.ones(train_edge_label_index.size(1))

        train_data = Data(
            x=data.x,
            edge_label_index=train_edge_label_index,
            edge_label=train_edge_label,
            x_additional=data.x_additional
        )

        train_val_sets.append((train_data, val_data))

    return train_val_sets


def split_tfs(data, tfs_index):
    # 如果张量在CUDA上，先转移到CPU上
    if data.edge_index.is_cuda:
        data.edge_index = data.edge_index.cpu()

    # 获取边的索引和对应的节点
    edges = data.edge_index.t().numpy()
    num_edges = edges.shape[0]

    train_val_sets = []

    for tf_node in tfs_index:
        # 查找边的第一个元素在 tf_node 中的边
        val_mask = np.isin(edges[:, 0], tf_node)
        val_edges = edges[val_mask]

        # 将 val_edges 转为 tensor 格式
        val_edge_index = torch.tensor(val_edges.T, dtype=torch.long)

        # 为验证集进行负采样，生成等量的负样本
        num_positive_samples = val_edge_index.size(1)
        num_neg_samples = num_positive_samples
        neg_samples = negative_sampling(edge_index=data.edge_index, num_nodes=data.x.size(0), num_neg_samples=num_neg_samples)

        # 构造 val_data
        val_edge_label_index = torch.cat([val_edge_index, neg_samples], dim=1)
        val_edge_label = torch.cat([torch.ones(num_positive_samples), torch.zeros(num_neg_samples)], dim=0)

        val_data = Data(
            x=data.x, 
            edge_index=val_edge_index, 
            edge_label_index=val_edge_label_index, 
            edge_label=val_edge_label, 
            x_additional=data.x_additional
        )

        # 确定训练集的大小
        num_train_edges = max(1, int(0.1 * num_edges))  # 至少选择1条边

        # 从原始数据中排除 val_data 中的正向边
        remaining_edges = edges[~val_mask]

        # 在排除后的边中随机选取边，直到满足 num_train_edges
        selected_train_edges = remaining_edges[np.random.choice(remaining_edges.shape[0], num_train_edges, replace=False)]

        # 将 selected_train_edges 转为 tensor 格式
        train_edge_index = torch.tensor(selected_train_edges.T, dtype=torch.long)

        # 从 train_data 中删除50%的边
        num_edges_to_remove = max(1, int(0.5 * train_edge_index.size(1)))  # 至少删除1条边
        remaining_indices = np.random.choice(train_edge_index.size(1), train_edge_index.size(1) - num_edges_to_remove, replace=False)
        train_edge_index = train_edge_index[:, remaining_indices]

        # 构造 edge_label_index 和 edge_label
        train_edge_label_index = train_edge_index
        train_edge_label = torch.ones(train_edge_label_index.size(1))

        # 构造 train_data
        train_data = Data(
            x=data.x,
            edge_index=train_edge_index,
            edge_label_index=train_edge_label_index,
            edge_label=train_edge_label,
            x_additional=data.x_additional
        )

        # 保存 train_data 和 val_data
        train_val_sets.append((train_data, val_data))

    return train_val_sets



def generate_train_val_kfold(data, num_splits=10):
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    indices = np.random.permutation(num_edges)
    split_edges = np.array_split(indices, num_splits)
    train_val_sets = []

    device = data.x.device  # 获取数据的设备

    for i in range(num_splits):
        # 验证数据
        val_edges = np.concatenate([split_edges[j] for j in range(num_splits) if j != i], axis=0)
        val_edge_index = edge_index[:, val_edges].to(device)

        # 生成负样本
        neg_edges = []
        while len(neg_edges) < len(val_edges):
            i_neg, j_neg = np.random.randint(0, data.x.size(0), 2)
            if (i_neg != j_neg) and (not ((edge_index[0] == i_neg) & (edge_index[1] == j_neg)).any()) and (not ((edge_index[0] == j_neg) & (edge_index[1] == i_neg)).any()):
                neg_edges.append([i_neg, j_neg])
        neg_edge_index = torch.tensor(neg_edges).t().to(device)

        # 合并正负样本
        val_edge_label_index = torch.cat([val_edge_index, neg_edge_index], dim=1)
        val_edge_label = torch.cat([torch.ones(val_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))]).to(device)

        # 训练数据
        train_edges = split_edges[i]
        train_edge_index = edge_index[:, train_edges].to(device)

        # 从训练数据中随机删除50%的边
        #train_indices = np.random.permutation(train_edge_index.size(1))
        #train_indices = train_indices[:train_edge_index.size(1) // 2]
        #train_edge_index = train_edge_index[:, train_indices]

        # 生成训练数据的标签
        train_edge_label_index = train_edge_index
        train_edge_label = torch.ones(train_edge_index.size(1)).to(device)

        # 创建PyTorch Geometric Data对象
        val_data = Data(x=data.x, edge_label_index=val_edge_label_index, edge_label=val_edge_label, x_additional=data.x_additional)
        train_data = Data(x=data.x, edge_label_index=train_edge_label_index, edge_label=train_edge_label, x_additional=data.x_additional)

        # 存放结果
        train_val_sets.append((train_data, val_data))

    return train_val_sets



import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data

def split_edges(data, num_train_edges, num_sets):
    train_val_sets = []

    for _ in range(num_sets):
        edge_index = data.edge_index
        num_edges = edge_index.size(1)

        # 随机选择训练集的边
        perm = torch.randperm(num_edges)
        train_edges = perm[:num_train_edges]
        val_edges = perm[num_train_edges:]

        # 创建训练集
        train_edge_index = edge_index[:, train_edges]
        
        # 随机删除一半的训练集边
        num_train_edges = train_edge_index.size(1)
        perm = torch.randperm(num_train_edges)
        num_to_keep = num_train_edges // 2
        train_edge_index = train_edge_index[:, perm[:num_to_keep]]

        # 创建验证集
        val_edge_index = edge_index[:, val_edges]

        # 为验证集生成等量的负样本
        num_val_edges = val_edge_index.size(1)
        neg_edge_index = negative_sampling(edge_index, num_nodes=data.x.size(0), num_neg_samples=num_val_edges)

        # 合并正负样本到val_data.edge_label_index
        val_edge_label_index = torch.cat([val_edge_index, neg_edge_index], dim=1)

        # 标签：正样本为1，负样本为0
        val_edge_label = torch.cat([torch.ones(num_val_edges), torch.zeros(num_val_edges)])

        # 创建 train_data 和 val_data
        train_data = Data(
            x=data.x,
            edge_index=train_edge_index,
            x_additional=data.x_additional if 'x_additional' in data else None
        )

        val_data = Data(
            x=data.x,
            edge_index=edge_index,
            x_additional=data.x_additional if 'x_additional' in data else None,
            edge_label_index=val_edge_label_index,
            edge_label=val_edge_label
        )

        train_val_sets.append((train_data, val_data))

    return train_val_sets

