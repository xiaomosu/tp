import torch
import torch.nn as nn

class SampleWeightedLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=2, eps=1e-8):
        super(SampleWeightedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, inputs, targets):
        """
        inputs: Tensor of model outputs, shape [batch_size, num_classes] for binary classification.
        targets: Tensor of true labels, shape [batch_size]
        """
        # 对于二分类任务，inputs 是 [batch_size, 2]，我们只需要选择第二列（正类的概率）
        p = torch.sigmoid(inputs[:, 1])  # 选择正类的概率
        y = targets.float()
        # print(f"形状：{p.shape},{y.shape}")

        # 确保 y 的形状是 [batch_size]，如果 y 是 [batch_size, 1]，则 squeeze 成 [batch_size]
        y = y.squeeze()

        # 计算每个样本的权重
        weight = self.alpha * (1 - p)**self.beta * y + (1 - self.alpha) * p**self.beta * (1 - y)

        # 计算损失
        loss = -(y * torch.log(p + self.eps) + (1 - y) * torch.log(1 - p + self.eps))

        return (weight * loss).mean()

