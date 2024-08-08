import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 初始化空列表用于存储数据
data_auc = []
data_roc = []

# 定义文件路径和文件名模板
base_path = '/Users/zfd297/work/scmte_validate/ablation/gene/'
file_template = '{}.csv'

# 读取每个CSV文件并提取数据
for i in range(10):
    file_path = os.path.join(base_path, file_template.format(i))
    df = pd.read_csv(file_path)
    data_auc.append(df['AUC'].values)
    data_roc.append(df['ROC'].values)

# 将数据转换为DataFrame以便于绘图
data_auc_df = pd.DataFrame(data_auc).T
data_roc_df = pd.DataFrame(data_roc).T

# 合并数据以便于绘图
data_combined = pd.concat([data_auc_df, data_roc_df], axis=1)
data_combined.columns = [f'Experiment {i} AUC' for i in range(10)] + [f'Experiment {i} ROC' for i in range(10)]

# 调整数据顺序使AUC和ROC交替排列
box_data = []
labels = []
positions = []
pos = 1
for i in range(10):
    box_data.append(data_combined[f'Experiment {i} AUC'])
    box_data.append(data_combined[f'Experiment {i} ROC'])
    labels.append(f'{i} AUC')
    labels.append(f'{i} ROC')
    positions.append(pos)
    positions.append(pos + 0.5)
    pos += 2

# 计算均值
means_auc = [np.mean(data_combined[f'Experiment {i} AUC']) for i in range(10)]
means_roc = [np.mean(data_combined[f'Experiment {i} ROC']) for i in range(10)]

# 绘制箱线图
plt.figure(figsize=(14, 7))

box = plt.boxplot(box_data, positions=positions, widths=0.4, patch_artist=True)

# 设置箱线图颜色
colors = ['purple', 'skyblue'] * 10
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# 绘制趋势线
mean_positions_auc = [i for i in range(1, 20, 2)]
mean_positions_roc = [i + 0.5 for i in range(1, 20, 2)]
plt.plot(mean_positions_auc, means_auc, color='purple', marker='o', linestyle='-', linewidth=2, markersize=5, label='AUROC')
plt.plot(mean_positions_roc, means_roc, color='skyblue', marker='o', linestyle='-', linewidth=2, markersize=5, label='AUPRC')

# 添加图例并调整大小
plt.legend(fontsize=14)

# 设置标题和标签
plt.title('', fontsize=16, fontweight='bold')
plt.xlabel('', fontsize=16, fontweight='bold')
plt.ylabel('', fontsize=16, fontweight='bold')

# 设置x轴刻度
plt.xticks(ticks=[i + 0.25 for i in range(1, 20, 2)], labels=[f'{i/10:.1f}' for i in range(10)], rotation=45, fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

# 保存图像
plt.tight_layout()
plt.savefig('/Users/zfd297/work/scmte_validate/ablation/gene/auc_roc_boxplot.png', dpi=300)

# 显示图像
plt.show()
