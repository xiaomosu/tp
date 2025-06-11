import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm
from models.model_new import *
import time
import os
from datetime import datetime

# 数据加载与预处理
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['CONS_NO'])
    train_df, test_df = train_test_split(df, test_size=0.4, random_state=66, stratify=df['FLAG'])
    X_train = train_df.drop(columns=['FLAG'])
    y_train = train_df["FLAG"]
    X_test = test_df.drop(columns=['FLAG'])
    y_test = test_df["FLAG"]
    X_train_np = np.array(X_train).reshape(X_train.shape[0], 1, X_train.shape[1])
    y_train_np = np.array(y_train).reshape(y_train.shape[0], 1)
    X_test_np = np.array(X_test).reshape(X_test.shape[0], 1, X_test.shape[1])
    y_test_np = np.array(y_test).reshape(y_test.shape[0], 1)
    X_train_ts = torch.from_numpy(X_train_np).float()
    y_train_ts = torch.from_numpy(y_train_np).float()
    X_test_ts = torch.from_numpy(X_test_np).float()
    y_test_ts = torch.from_numpy(y_test_np).float()
    train_set = Data.TensorDataset(X_train_ts, y_train_ts)
    test_set = Data.TensorDataset(X_test_ts, y_test_ts)
    return train_set, test_set

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

# 训练函数
def train(model, optimizer, train_loader, epoch=5):
    model.train()
    epoch_loss = []
    for e in range(epoch):
        batch_loss = []
        with tqdm(train_loader, desc=f"训练 Epoch {e+1}/{epoch}") as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                labels = target.squeeze().long()
                loss = model.lossfunction(output, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                pbar.set_postfix(loss=loss.item())
        epoch_loss.append(sum(batch_loss) / (len(batch_loss) * batch_size))
    return sum(epoch_loss) / len(epoch_loss)

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
                data, target = data.cuda(), target.cuda()
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
    df = pd.DataFrame(metrics_history)
    os.makedirs('map', exist_ok=True)
    filepath = os.path.join('map', filename)
    df.to_csv(filepath, index=False)
    print(f"指标已保存到: {filepath}")

# 主函数
def main():
    global batch_size
    batch_size = 64
    epochs = 20
    learning_rate = 0.0001
    train_set, test_set = load_and_preprocess_data("./dataset/data_del.csv")
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model = CNN_Attention_WithAttention().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_times = []
    test_times = []
    metrics_history = []

    for epoch in range(epochs):
        print(f"\n{'='*20} Epoch {epoch + 1}/{epochs} {'='*20}")
        train_start = time.time()
        train_loss = train(model, optimizer, train_loader, epoch=1)
        train_end = time.time()
        train_duration = train_end - train_start
        train_times.append(train_duration)
        test_start = time.time()
        metrics = test(model, test_loader)
        test_end = time.time()
        test_duration = test_end - test_start
        test_times.append(test_duration)

        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': metrics['loss'],
            'accuracy': metrics['accuracy'],
            'auc': metrics['auc'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'train_time': train_duration,
            'test_time': test_duration,
            'map100': metrics['map100'],
            'map200': metrics['map200']
        }
        metrics_history.append(epoch_metrics)

        print(f"训练损失: {train_loss:.4f} | 训练时间: {train_duration:.2f}秒")
        print(f"测试损失: {metrics['loss']:.4f} | 准确率: {metrics['accuracy']:.4f} | AUC: {metrics['auc']:.4f}")
        print(f"精确率: {metrics['precision']:.4f} | 召回率: {metrics['recall']:.4f} | F1分数: {metrics['f1']:.4f}")
        print(f"测试时间: {test_duration:.2f}秒")
        print(f"mAP@100: {metrics['map100']:.4f} | mAP@200: {metrics['map200']:.4f}")

    avg_train_time = sum(train_times) / len(train_times)
    avg_test_time = sum(test_times) / len(test_times)
    print("\n训练和评估完成!")
    print(f"每轮平均训练时间: {avg_train_time:.2f}秒")
    print(f"每轮平均测试时间: {avg_test_time:.2f}秒")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_metrics_to_csv(metrics_history, f'model_metrics_{timestamp}.csv')

# 程序入口
if __name__ == "__main__":
    main()
