# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_geometric.utils import negative_sampling
# from sklearn.metrics import roc_auc_score, average_precision_score

# class TransformerLP(nn.Module):
#     def __init__(self, in_channels, additional_channels, hidden_channels, num_heads=8, dropout=0.7):
#         super().__init__()
#         self.embedding = nn.Linear(in_channels + additional_channels, hidden_channels)
#         self.self_attention = nn.MultiheadAttention(hidden_channels, num_heads, dropout=dropout)
#         self.norm1 = nn.LayerNorm(hidden_channels)
#         self.norm2 = nn.LayerNorm(hidden_channels)
#         self.feedforward = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels * 4),
#             nn.ReLU(),
#             nn.Linear(hidden_channels * 4, hidden_channels)
#         )
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(hidden_channels * 2, 1)

#     def encode(self, x, x_additional):
#         x = torch.cat([x, x_additional], dim=1)
#         x = self.embedding(x)
#         x = x.unsqueeze(0)
#         attn_output, _ = self.self_attention(x, x, x)
#         x = x + self.dropout(attn_output)
#         x = self.norm1(x)
#         ff_output = self.feedforward(x)
#         x = x + self.dropout(ff_output)
#         x = self.norm2(x)
#         return x.squeeze(0)

#     def decode(self, z, edge_label_index):
#         src = z[edge_label_index[0]]
#         dst = z[edge_label_index[1]]
#         x = torch.cat([src, dst], dim=1)
#         return self.fc(x).squeeze(-1)

#     def forward(self, x, x_additional, edge_label_index):
#         z = self.encode(x, x_additional)
#         return self.decode(z, edge_label_index)

# def negative_sample(data):
#     neg_edge_index = negative_sampling(
#         edge_index=data.edge_label_index, num_nodes=data.x.size(0),
#         num_neg_samples=data.edge_label_index.size(1), method='sparse')
    
#     edge_label_index = torch.cat([data.edge_label_index, neg_edge_index], dim=-1)
#     edge_label = torch.cat([
#         data.edge_label,
#         data.edge_label.new_zeros(neg_edge_index.size(1))
#     ], dim=0)

#     return edge_label, edge_label_index


# def train_model(train_data, val_data, hidden_channels=64, num_heads=8, dropout=0.6, 
#                 lr=0.00001, weight_decay=5e-4, num_epochs=150, print_interval=10):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = TransformerLP(train_data.x.size(1), train_data.x_additional.size(1), 
#                           hidden_channels, num_heads, dropout).to(device)
#     optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
#     criterion = nn.BCEWithLogitsLoss().to(device)
    
#     train_data = train_data.to(device)
#     val_data = val_data.to(device)
    
#     edge_label, edge_label_index = negative_sample(train_data)
    
#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()
#         out = model(train_data.x, train_data.x_additional, edge_label_index)
#         loss = criterion(out, edge_label)
#         loss.backward()
#         optimizer.step()
        
#         if (epoch + 1) % print_interval == 0:
#             val_auc, val_roc = test_model(model, val_data)
#             print(f"Epoch {epoch + 1:03d}, Loss: {loss.item():.4f}, "
#                   f"Validation AUPRC: {val_auc:.4f}, Validation AUROC: {val_roc:.4f}")

#     # 训练结束后进行最后一次验证
#     final_val_auc, final_val_roc = test_model(model, val_data)
#     print(f"Final - Validation AUPRC: {final_val_auc:.4f}, Validation AUROC: {final_val_roc:.4f}")

#     return model, final_val_auc, final_val_roc



# def test_model(model, data):
#     device = next(model.parameters()).device
#     data = data.to(device)
#     model.eval()
#     with torch.no_grad():
#         z = model.encode(data.x, data.x_additional)
#         out = model.decode(z, data.edge_label_index).sigmoid()
    
#     auprc = average_precision_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
#     auroc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    
#     return auprc, auroc

# # 如果需要直接运行此文件进行测试，可以添加以下代码
# if __name__ == "__main__":
#     # 这里可以添加测试代码
#     pass




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

class TransformerLP(nn.Module):
    def __init__(self, in_channels, additional_channels, hidden_channels, num_heads=8, dropout=0.7):
        super().__init__()
        self.attention = nn.Linear(in_channels + additional_channels, 1)
        self.embedding = nn.Linear(in_channels + additional_channels, hidden_channels)
        self.self_attention = nn.MultiheadAttention(hidden_channels, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.ReLU(),
            nn.Linear(hidden_channels * 4, hidden_channels)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels * 2, 1)

    def encode(self, x, x_additional):
        combined = torch.cat([x, x_additional], dim=1)
        attention_scores = self.attention(combined)
        attention_weights = F.softmax(attention_scores, dim=1)
        attended_features = attention_weights * combined
        
        x = self.embedding(attended_features)
        x = x.unsqueeze(0)
        attn_output, _ = self.self_attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.feedforward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x.squeeze(0)

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        x = torch.cat([src, dst], dim=1)
        return self.fc(x).squeeze(-1)

    def forward(self, x, x_additional, edge_label_index):
        z = self.encode(x, x_additional)
        return self.decode(z, edge_label_index)

def negative_sample(data, bl):
    neg_edge_index = negative_sampling(
        edge_index=data.edge_label_index, num_nodes=data.x.size(0),
        num_neg_samples=data.edge_label_index.size(1)*bl, method='sparse')
    
    edge_label_index = torch.cat([data.edge_label_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([
        data.edge_label,
        data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    return edge_label, edge_label_index

def train_model(train_data, hidden_channels=64, num_heads=8, dropout=0.6, 
                lr=0.00001, weight_decay=5e-4, num_epochs=150, print_interval=10,bl=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerLP(train_data.x.size(1), train_data.x_additional.size(1), 
                          hidden_channels, num_heads, dropout).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    train_data = train_data.to(device)
    #val_data = val_data.to(device)
    
    #edge_label, edge_label_index = negative_sample(train_data)
    
    for epoch in range(num_epochs):
        edge_label, edge_label_index = negative_sample(train_data,bl)
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.x_additional, edge_label_index)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
        
        # if (epoch + 1) % print_interval == 0:
        #     val_auc, val_roc = test_model(model, val_data)
        #     print(f"Epoch {epoch + 1:03d}, Loss: {loss.item():.4f}, "
        #         f"Validation AUPRC: {val_auc:.4f}, Validation AUROC: {val_roc:.4f}")

    #final_val_auc, final_val_roc = test_model(model, val_data)
    #print(f"Final - Validation AUPRC: {final_val_auc:.4f}, Validation AUROC: {final_val_roc:.4f}")

    return model
#, final_val_auc, final_val_roc

def test_model(model, data):
    device = next(model.parameters()).device
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.x_additional)
        out = model.decode(z, data.edge_label_index).sigmoid()
    
    auprc = average_precision_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    auroc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    
    return auprc, auroc

if __name__ == "__main__":
    # 这里可以添加测试代码
    pass