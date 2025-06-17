import os
import random
from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
torch.backends.cudnn.benchmark=True
# from models.model_old import *
# from models.model import *
from models.model_used import *
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.utils.data as Data
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
# 设置GPU设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 默认选择GPU 0，或者CPU

# 从CSV文件加载数据集
df = pd.read_csv("./dataset/data_del.csv")

# 删除无用列"CONS_NO"
df = df.drop(columns=['CONS_NO', ])
df.tail()

# 将数据集分为异常样本和正常样本
anomalies = df[df["FLAG"] == 1]
normal = df[df["FLAG"] == 0]

print(anomalies.shape, normal.shape)

# 将数据集划分为训练集和测试集，测试集比例为30%
DF_train, DF_test = train_test_split(df, test_size=0.3, random_state=66)
print(DF_train.shape, DF_test.shape)

# 将训练集和测试集分别分为特征（X）和标签（y）
DF_train_y = DF_train["FLAG"]
DF_train_X = DF_train.drop(columns=['FLAG'])

DF_test_y = DF_test["FLAG"]
DF_test_X = DF_test.drop(columns=['FLAG'])

# 将训练和测试数据转换为NumPy数组，并进行形状调整
train_y = np.array(DF_train_y).reshape(DF_train_y.shape[0], 1)
train_X = np.array(DF_train_X).reshape(DF_train_X.shape[0], 1, DF_train_X.shape[1])

test_y = np.array(DF_test_y).reshape(DF_test_y.shape[0], 1)
test_X = np.array(DF_test_X).reshape(DF_test_X.shape[0], 1, DF_test_X.shape[1])

print("训练数据集形状: train_X: %s 和 train_y: %s" % (train_X.shape, train_y.shape))
print("测试数据集形状: test_X: %s 和 test_y: %s" % (test_X.shape, test_y.shape))

# 将NumPy数组转换为PyTorch张量
train_y_ts = torch.from_numpy(train_y).float()
train_X_ts = torch.from_numpy(train_X).float()

test_y_ts = torch.from_numpy(test_y).float()
test_X_ts = torch.from_numpy(test_X).float()

# 创建PyTorch数据集
train_set = Data.TensorDataset(train_X_ts, train_y_ts)
test_set = Data.TensorDataset(test_X_ts, test_y_ts)

# 定义一些超参数
num_clients = 15  # 客户端数量
num_selected = 15  # 每轮选择的客户端数量
num_rounds = 20  # 训练轮次
epochs = 1  # 每轮客户端训练的epoch数
batch_size = 64  # 批量大小

# 设置随机种子以确保可重复性
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# 获取训练数据的标签
train_labels = train_y.squeeze()
positive_indices = np.where(train_labels == 1)[0]
negative_indices = np.where(train_labels == 0)[0]

print(f"原始数据集: 正样本比例 = {len(positive_indices)/len(train_labels):.4f}, 总样本数 = {len(train_labels)}")
print(f"正样本数: {len(positive_indices)}, 负样本数: {len(negative_indices)}")

# 计算每个客户端应获得的正负样本数量
pos_per_client = len(positive_indices) // num_clients
neg_per_client = len(negative_indices) // num_clients

# 分配客户端数据
client_datasets = []
remaining_pos = positive_indices.copy()
remaining_neg = negative_indices.copy()

for i in range(num_clients):
    # 最后一个客户端获取所有剩余数据
    if i == num_clients - 1:
        client_pos_indices = remaining_pos
        client_neg_indices = remaining_neg
    else:
        # 随机选择正样本索引
        if len(remaining_pos) >= pos_per_client:
            client_pos_indices = np.random.choice(remaining_pos, pos_per_client, replace=False)
            remaining_pos = np.setdiff1d(remaining_pos, client_pos_indices)
        else:
            client_pos_indices = remaining_pos
            remaining_pos = np.array([])
            
        # 随机选择负样本索引
        if len(remaining_neg) >= neg_per_client:
            client_neg_indices = np.random.choice(remaining_neg, neg_per_client, replace=False)
            remaining_neg = np.setdiff1d(remaining_neg, client_neg_indices)
        else:
            client_neg_indices = remaining_neg
            remaining_neg = np.array([])
    
    # 合并正负样本索引
    client_indices = np.concatenate([client_pos_indices, client_neg_indices])
    client_datasets.append(torch.utils.data.Subset(train_set, client_indices))
    
    # 打印此客户端的数据分布
    client_labels = train_labels[client_indices]
    pos_count = np.sum(client_labels == 1)
    total = len(client_indices)
    print(f"客户端 {i+1}: 正样本比例 = {pos_count/total:.4f}, 样本数量 = {total} (正样本: {pos_count}, 负样本: {total-pos_count})")
    print(f"客户端 {i+1} 分配后剩余正样本: {len(remaining_pos)}, 剩余负样本: {len(remaining_neg)}")


# 创建每个客户端的数据加载器
train_loader = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True, 
                                           worker_init_fn=lambda x: np.random.seed(random_seed+x)) for x in client_datasets]

# 创建测试数据加载器
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# 定义客户端更新函数
def client_update(client_model, optimizer, train_loader, epoch=5, device='cuda'):
    """
    客户端训练更新函数，训练客户端模型并返回损失值
    """
    client_model.train()
    epoch_loss = []
    for e in range(epoch):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)  # 将数据移到指定设备
            optimizer.zero_grad()
            output = client_model(data)

            labels = target.squeeze()
            labels = labels.long()
            loss = client_model.lossfunction(output, labels)

            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss) / (len(batch_loss) * batch_size))

    return sum(epoch_loss) / len(epoch_loss)

# 替换后的 mAP 计算方法
def precision_at_k(y_true, class_probs, k, threshold=0.5, class_of_interest=1, isSorted=False):
    if not isSorted:
        coi_probs = class_probs[:, class_of_interest]
        sorted_coi_probs = np.sort(coi_probs)[::-1]
        sorted_y = y_true[np.argsort(coi_probs)[::-1]]
    else:
        sorted_coi_probs = class_probs
        sorted_y = y_true
    sorted_coi_probs = sorted_coi_probs[:k]
    sorted_y = sorted_y[:k]
    sorted_predicted_classes = np.where(sorted_coi_probs > threshold, float(class_of_interest), 0.0)
    precisionK = np.sum(sorted_predicted_classes == sorted_y) / k
    return precisionK

def map_at_N(y_true, class_probs, N, thrs=0.5, class_of_interest=1):
    pks = []
    coi_probs = class_probs[:, class_of_interest]
    sorted_coi_probs = np.sort(coi_probs)[::-1]
    sorted_y = y_true[np.argsort(coi_probs)[::-1]]
    sorted_coi_probs = sorted_coi_probs[:N]
    sorted_y = sorted_y[:N]
    top_coi_indexes = np.argwhere(sorted_y > 0)
    for value in top_coi_indexes:
        limite = value[0] + 1
        pks.append(
            precision_at_k(sorted_y[:limite], sorted_coi_probs[:limite],
                           limite, threshold=thrs, isSorted=True)
        )
    pks = np.array(pks)
    return pks.mean() if len(pks) > 0 else 0.0

# 定义服务器聚合函数
def server_aggregate(global_model, save_folder="model0/", batch_size=5):
    """
    聚合客户端的模型参数，更新全局模型
    聚合操作从文件夹加载模型，每次批量加载并聚合
    """
    print("开始聚合")
    global_dict = global_model.state_dict()

    # 获取文件夹中所有模型文件
    model_files = [f for f in os.listdir(save_folder) if f.endswith(".pth")]
    total_files = len(model_files)
    
    # 定义当前处理的文件索引
    current_index = 0

    # 分批次处理客户端模型
    while current_index < total_files:
        # 每次加载 batch_size 个模型
        batch_files = model_files[current_index:current_index + batch_size]
        client_models = []

        for filename in batch_files:
            file_path = os.path.join(save_folder, filename)
            client_model = STAFN().to(device)  # 假设使用相同的模型
            client_model.load_state_dict(torch.load(file_path))
            client_models.append(client_model)

     # 聚合客户端模型
        for k in global_dict.keys():
            w0 = global_dict[k].to(device)  # 将全局模型的权重移至 GPU
            x5 = torch.zeros_like(w0).to(device)  # 初始化累加变量，并确保它在 GPU 上
    
            # 对每个客户端的模型进行差异计算并累加
            for i in range(len(client_models)):
                client_model = client_models[i].state_dict()[k].to(device)  # 获取客户端模型的权重并保留在 GPU 上
                x1 = w0 - client_model.float()
                x2 = x1 * 1024
                x3 = torch.clamp(x2, min=-32768, max=32767)
                x4 = torch.div(x3, 1)
                x5 = x5 + x4  # 累加差异
    
            # 对累加结果进行平均
            x6 = torch.div(x5, 1024 * len(client_models))
            x7 = w0 - x6  # 计算聚合结果
            
            # 更新全局模型
            global_dict[k] = x7
            
        # 将更新后的全局模型加载回 GPU
        global_model.load_state_dict(global_dict)
        # 同步所有客户端的模型
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

        # 清理文件夹中已处理的模型
        for filename in batch_files:
            file_path = os.path.join(save_folder, filename)
            os.remove(file_path)
        
        print(f"已处理并清理批次：{current_index} 到 {current_index + len(batch_files) - 1}")
        
        # 更新文件索引，准备处理下一个批次
        current_index += len(batch_files)

    print(f"文件夹 {save_folder} 已清空，准备下一轮训练。")

# 测试函数（已使用新的 mAP 计算方法）
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    all_pred_classes = []
    all_y_true = []
    all_y_scores = []

    with torch.no_grad():
        with tqdm(test_loader, desc="测试中") as pbar:
            for data, target in pbar:
                data, target = data.to(device), target.to(device)  # 将数据移到指定设备
                output = model(data)
                labels = target.squeeze().long()
                loss = model.lossfunction(output, labels)
                test_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred).long()).sum().item()
                all_labels.extend(target.cpu().numpy())
                all_preds.extend(output.softmax(dim=1)[:, 1].cpu().numpy())
                all_pred_classes.extend(pred.cpu().numpy())
                all_y_true.extend(target.cpu().numpy())
                all_y_scores.extend(output.softmax(dim=1)[:, 1].cpu().numpy())

    all_y_true = np.array(all_y_true).flatten()
    all_y_scores = np.array(all_y_scores).flatten()

    # 构造class_probs格式用于 map_at_N
    class_probs = np.zeros((len(all_y_scores), 2))
    class_probs[:, 1] = all_y_scores
    class_probs[:, 0] = 1 - all_y_scores

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    auc = roc_auc_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_pred_classes)
    recall = recall_score(all_labels, all_pred_classes)
    f1 = f1_score(all_labels, all_pred_classes)
    cm = confusion_matrix(all_labels, all_pred_classes)

    # 使用新方法计算mAP
    map100 = map_at_N(all_y_true, class_probs, 100)
    map200 = map_at_N(all_y_true, class_probs, 200)

    return {
        'loss': test_loss,
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'map100': map100,
        'map200': map200
    }

# 保存结果到CSV文件
def save_metrics_to_csv(metrics_history, filename):
    """将训练过程中的指标保存到CSV文件"""
    # 创建DataFrame
    df = pd.DataFrame(metrics_history)
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 保存到CSV
    filepath = os.path.join('results', filename)
    df.to_csv(filepath, index=False)
    print(f"指标已保存到: {filepath}")

# 初始化全局模型
global_model = STAFN().to(device)

# 初始化指标记录
metrics_history = []
best_acc = 0
best_auc = 0
best_f1 = 0

# 每轮训练分批进行，最多同时加载5个客户端模型
batch_size = 5  # 每次最多训练5个客户端
num_batches = (num_selected + batch_size - 1) // batch_size  # 计算批次数，确保完全覆盖所有客户端

for r in range(num_rounds):
    # 随机选择客户端
    client_idx = np.random.permutation(num_clients)[:num_selected]
    loss = 0

    # 逐批加载客户端模型
    for batch in range(num_batches):
        # 选择当前批次的客户端
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_selected)
        batch_client_idx = client_idx[start_idx:end_idx]

        # 初始化当前批次的客户端模型并同步全局模型
        current_client_models = [STAFN().to(device) for _ in range(len(batch_client_idx))]
        for i, model in enumerate(current_client_models):
            model.load_state_dict(global_model.state_dict())  # 同步全局模型

        # 定义优化器
        current_opt = [optim.Adam(model.parameters(), lr=0.0001) for model in current_client_models]

        # 对当前批次的客户端进行训练
        batch_loss = 0
        for i in tqdm(range(len(batch_client_idx)), leave=True):
            batch_loss += client_update(current_client_models[i], current_opt[i], train_loader[client_idx[i]], epoch=epochs)

        # 计算当前批次的平均训练损失
        avg_train_loss = batch_loss / len(batch_client_idx)

        # 直接保存当前批次的客户端模型到文件夹中
        save_folder = "model0/"  # 保存客户端模型的文件夹
        os.makedirs(save_folder, exist_ok=True)
        for i, client_model in enumerate(current_client_models):
            model_path = os.path.join(save_folder, f"client_model_{start_idx + i}.pth")
            torch.save(client_model.state_dict(), model_path)  # 保存客户端模型到文件夹

    # 在这一轮结束后进行一次聚合
    server_aggregate(global_model, save_folder="model0/")  # 使用保存的模型进行聚合

    # 在测试集上评估全局模型
    metrics = test(global_model, test_loader)

    # 记录当前轮次的指标
    round_metrics = {
        'round': r + 1,
        'train_loss': avg_train_loss,
        'test_loss': metrics['loss'],
        'accuracy': metrics['accuracy'],
        'auc': metrics['auc'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'map100': metrics['map100'],
        'map200': metrics['map200']
    }
    metrics_history.append(round_metrics)

    # 打印当前轮次的指标
    print(f'第 {r+1} 轮:')
    print(f'平均训练损失: {avg_train_loss:.4f} | 测试损失: {metrics["loss"]:.4f}')
    print(f'准确率: {metrics["accuracy"]:.4f} | AUC: {metrics["auc"]:.4f}')
    print(f'精确率: {metrics["precision"]:.4f} | 召回率: {metrics["recall"]:.4f} | F1分数: {metrics["f1"]:.4f}')
    print(f"mAP@100: {metrics['map100']:.4f} | mAP@200: {metrics['map200']:.4f}")

    # 记录最佳指标
    if metrics['accuracy'] > best_acc:
        best_acc = metrics['accuracy']
        print(f"新的最高准确率: {best_acc:.4f}")

    if metrics['auc'] > best_auc:
        best_auc = metrics['auc']
        print(f"新的最高AUC: {best_auc:.4f}")

    if metrics['f1'] > best_f1:
        best_f1 = metrics['f1']
        print(f"新的最高F1分数: {best_f1:.4f}")

# 在训练结束后保存所有指标
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_metrics_to_csv(metrics_history, f'stafn_{timestamp}.csv')

# 在所有轮次训练结束后打印最终的最高指标
print("\n训练完成!")
print(f"最高准确率: {best_acc:.4f}")
print(f"最高AUC: {best_auc:.4f}")
print(f"最高F1分数: {best_f1:.4f}")
