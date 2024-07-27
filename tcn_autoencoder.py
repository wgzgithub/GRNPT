import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# 定义TCN模块
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 定义TCN自编码器
class TCNAutoencoder(nn.Module):
    def __init__(self, input_size, encoder_channels, decoder_channels, dropout=0.1):
        super(TCNAutoencoder, self).__init__()
        self.encoder = TemporalConvNet(input_size, encoder_channels, dropout=dropout)
        self.decoder = TemporalConvNet(encoder_channels[-1], decoder_channels, dropout=dropout)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def train_and_extract_features(X_norm, input_size=1, encoder_channels=[36, 36, 36], 
                               decoder_channels=[36, 36, 36, 1], num_epochs=200, 
                               batch_size=64, learning_rate=0.0001, weight_decay=1e-6):
    # 转换数据为 PyTorch 张量
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(-1).transpose(1, 2)

    # 创建 TensorDataset 和 DataLoader
    dataset = TensorDataset(X_tensor, X_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNAutoencoder(input_size, encoder_channels, decoder_channels, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 使用学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # 用于记录训练过程中的损失值
    train_losses = []

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for seq, _ in data_loader:
            seq = seq.to(device)
            output = model(seq)
            loss = criterion(output, seq)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        average_train_loss = epoch_train_loss / len(data_loader)
        train_losses.append(average_train_loss)

        if epoch % 5 == 0:  # 每5个epoch打印一次损失
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}')
            scheduler.step(average_train_loss)
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}')

    print("Training complete.")

    # # 可视化训练损失
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Over Time')
    # plt.legend()
    # plt.show()

    # 生成时间序列特征
    all_dataset = TensorDataset(X_tensor, X_tensor)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)

    features = []
    model.eval()
    with torch.no_grad():
        for seq, _ in all_loader:
            seq = seq.to(device)
            latent = model.encoder(seq)
            features.append(latent.cpu())

    features = torch.cat(features, dim=0)
    print("Features extracted. Shape:", features.shape)

    # 使用最大池化方法降维
    reduced_features = features.max(dim=2).values
    print("Reduced features shape:", reduced_features.shape)

    return model, reduced_features

# 使用示例
if __name__ == "__main__":
    # 假设 X_norm 是已经准备好的归一化数据
    # X_norm = ...  # 用户提供的数据

    model, features = train_and_extract_features(X_norm)
    print("Feature extraction complete.")