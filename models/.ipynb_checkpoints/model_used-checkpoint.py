import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(ChannelAttentionModule, self).__init__()
        # 使用全连接层来计算通道注意力权重
        self.fc1 = nn.Linear(input_dim, input_dim // 2)  # 降维
        self.fc2 = nn.Linear(input_dim // 2, input_dim)  # 恢复维度

    def forward(self, x):
        # x 的形状为 (batch_size, channels, seq_len)
        # 计算通道注意力分数
        channel_attn = torch.mean(x, dim=-1)  # 对序列维度求平均，得到每个通道的表示
        channel_attn = F.relu(self.fc1(channel_attn))  # 降维
        channel_attn = torch.sigmoid(self.fc2(channel_attn))  # 恢复维度并进行归一化

        # 将通道注意力权重应用到输入特征上
        # 结果是每个通道都有一个注意力权重，x * channel_attn 实现了加权
        channel_attn = channel_attn.unsqueeze(-1)  # 增加最后一维与seq_len对齐
        return x * channel_attn  # 加权后的特征

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # 残差连接


class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        # 定义查询、键、值的权重矩阵
        self.query_weight = nn.Parameter(torch.randn(input_dim, input_dim))  # (n_features, n_features)
        self.key_weight = nn.Parameter(torch.randn(input_dim, input_dim))  # (n_features, n_features)
        self.value_weight = nn.Parameter(torch.randn(input_dim, input_dim))  # (n_features, n_features)

    def forward(self, x):     
        # x 的形状为 (batch_size, seq_len, n_features)
        # # print(f"Input shape to AttentionModule: {x.shape}")

        # 计算查询、键、值
        query = torch.matmul(x, self.query_weight)  # (batch_size, seq_len, n_features)
        key = torch.matmul(x, self.key_weight)      # (batch_size, seq_len, n_features)
        value = torch.matmul(x, self.value_weight)  # (batch_size, seq_len, n_features)

        # 打印计算后的查询、键和值的形状
        # # print(f"Query shape: {query.shape}")
        # # print(f"Key shape: {key.shape}")
        # # print(f"Value shape: {value.shape}")

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, seq_len, seq_len)
        # # print(f"Attention scores shape: {attention_scores.shape}")

        # 添加scale操作
        d_k = query.size(-1)  # 获取key向量的维度
        attention_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # 归一化处理，得到每个时间步的注意力分数
        attention_scores = F.softmax(attention_scores, dim=-1)  # Softmax 对注意力分数进行归一化
        # # print(f"Attention scores after softmax shape: {attention_scores.shape}")

        # 对每个时间步的每个特征进行加权求和
        weighted_output = torch.matmul(attention_scores, value)  # (batch_size, seq_len, n_features)
        # # print(f"Weighted output shape: {weighted_output.shape}")

        return weighted_output

class tcn_cnn_selfattention(nn.Module):
    def __init__(self, seq_len=1035):
        super().__init__()

        # TCN模块，三层 TCN
        self.tcn1 = TemporalBlock(1, 64, 3, 1, 1, 1, 0.05)   # 时间步长为1
        self.tcn2 = TemporalBlock(64, 128, 3, 1, 7, 7, 0.05)  # 时间步长为7

        # Branch 2: CNN 特征提取器
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        # Attention模块: 为输入数据加上自注意力机制
        self.attention = AttentionModule(1)  # 自注意力机制直接作用于输入数据

        # 融合: 在第二维度进行拼接
        self.fc1 = nn.Linear((128 * 2 +1)* seq_len, 128)  # 拼接后的特征维度，256*seq_len的四倍
        self.fc2 = nn.Linear(128, 2)  # 输出分类结果
        self.lossfunction = nn.CrossEntropyLoss()
        # 通道注意力机制
        self.channel_attention = ChannelAttentionModule(257)  # 拼接后的通道数（256+256）

    def forward(self, x):
        x1 = x.transpose(1, 2)  # 转置形状
        # 自注意力机制直接作用于输入数据
        attention_out = self.attention(x1)  # (batch_size, seq_len, n_features)
        attention_out = attention_out.transpose(1, 2)
        # print(f"Attention output shape: {attention_out.shape}")


        # TCN分支
        tcn_out1 = self.tcn1(x)  # (batch_size, seq_len, 64)
        # print(f"TCN output shape 1: {tcn_out1.shape}")
        
        tcn_out2 = self.tcn2(tcn_out1)  # (batch_size, seq_len, 128)
        # print(f"TCN output shape 2: {tcn_out2.shape}")
        
        # CNN分支
        cnn_out = F.relu(self.conv1(x))
        # print(f"CNN output shape 1: {cnn_out.shape}")
        
        cnn_out = F.relu(self.conv2(cnn_out))
        # print(f"CNN output shape 2: {cnn_out.shape}")
        

        # 将TCN、CNN和Attention的输出拼接起来
        x = torch.cat((tcn_out2, cnn_out, attention_out), dim=1)
        # print(f"Shape after concatenation: {x.shape}")

        x = self.channel_attention(x)

        # 展平拼接后的输出
        x = x.view(x.size(0), -1)
        # print(f"Shape after flattening: {x.shape}")

        # 全连接层
        x = F.relu(self.fc1(x))  # 第一全连接层
        # print(f"Shape after fc1: {x.shape}")
        
        output = self.fc2(x)  # 第二全连接层，输出分类结果
        # print(f"Output shape: {output.shape}")
        
        return output

#消融实验
class TCNOnly(nn.Module):
    def __init__(self, seq_len=1035):
        super(TCNOnly, self).__init__()
        self.tcn1 = TemporalBlock(1, 64, 3, 1, 1, 1, 0.05)
        self.tcn2 = TemporalBlock(64, 128, 3, 1, 7, 7, 0.05)
        self.fc1 = nn.Linear(128 * seq_len, 128)
        self.fc2 = nn.Linear(128, 2)
        self.lossfunction = nn.CrossEntropyLoss()
    
    def forward(self, x):
        tcn_out1 = self.tcn1(x)
        tcn_out2 = self.tcn2(tcn_out1)
        x = tcn_out2.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output


class CNNOnly(nn.Module):
    def __init__(self, seq_len=1035):
        super(CNNOnly, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * seq_len, 128)
        self.fc2 = nn.Linear(128, 2)
        self.lossfunction = nn.CrossEntropyLoss()
    
    def forward(self, x):
        cnn_out = F.relu(self.conv1(x))
        cnn_out = F.relu(self.conv2(cnn_out))
        x = cnn_out.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output

class AttentionOnly(nn.Module):
    def __init__(self, seq_len=1035):
        super(AttentionOnly, self).__init__()
        self.attention = AttentionModule(1)
        self.fc1 = nn.Linear(seq_len, 128)
        self.fc2 = nn.Linear(128, 2)
        self.lossfunction = nn.CrossEntropyLoss()
    
    def forward(self, x):
        attention_out = self.attention(x.transpose(1, 2))
        attention_out = attention_out.transpose(1, 2)
        x = attention_out.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output

class TCN_CNN(nn.Module):
    def __init__(self, seq_len=1035):
        super(TCN_CNN, self).__init__()
        self.tcn1 = TemporalBlock(1, 64, 3, 1, 1, 1, 0.05)
        self.tcn2 = TemporalBlock(64, 128, 3, 1, 7, 7, 0.05)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear((128 * 2) * seq_len, 128)
        self.fc2 = nn.Linear(128, 2)
        self.lossfunction = nn.CrossEntropyLoss()
    
    def forward(self, x):
        tcn_out1 = self.tcn1(x)
        tcn_out2 = self.tcn2(tcn_out1)
        cnn_out = F.relu(self.conv1(x))
        cnn_out = F.relu(self.conv2(cnn_out))
        x = torch.cat((tcn_out2, cnn_out), dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output

class TCN_Attention(nn.Module):
    def __init__(self, seq_len=1035):
        super(TCN_Attention, self).__init__()
        self.tcn1 = TemporalBlock(1, 64, 3, 1, 1, 1, 0.05)
        self.tcn2 = TemporalBlock(64, 128, 3, 1, 7, 7, 0.05)
        self.attention = AttentionModule(1)
        self.fc1 = nn.Linear((128 + 1) * seq_len, 128)
        self.fc2 = nn.Linear(128, 2)
        self.lossfunction = nn.CrossEntropyLoss()
    
    def forward(self, x):
        tcn_out1 = self.tcn1(x)
        tcn_out2 = self.tcn2(tcn_out1)
        attention_out = self.attention(x.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((tcn_out2, attention_out), dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output
        
class CNN_Attention(nn.Module):
    def __init__(self, seq_len=1035):
        super(CNN_Attention, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.attention = AttentionModule(1)
        self.fc1 = nn.Linear((128 + 1) * seq_len, 128)
        self.fc2 = nn.Linear(128, 2)
        self.lossfunction = nn.CrossEntropyLoss()
    
    def forward(self, x):
        cnn_out = F.relu(self.conv1(x))
        cnn_out = F.relu(self.conv2(cnn_out))
        attention_out = self.attention(x.transpose(1, 2)).transpose(1, 2)
        x = torch.cat((cnn_out, attention_out), dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output

#组合+通道注意力
class TCN_CNN_WithAttention(nn.Module):
    def __init__(self, seq_len=1035):
        super(TCN_CNN_WithAttention, self).__init__()
        self.tcn1 = TemporalBlock(1, 64, 3, 1, 1, 1, 0.05)
        self.tcn2 = TemporalBlock(64, 128, 3, 1, 7, 7, 0.05)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.channel_attention = ChannelAttentionModule(256)  # TCN(128) + CNN(128)
        self.fc1 = nn.Linear(256 * seq_len, 128)
        self.fc2 = nn.Linear(128, 2)
        self.lossfunction = nn.CrossEntropyLoss()
    
    def forward(self, x):
        tcn_out1 = self.tcn1(x)
        tcn_out2 = self.tcn2(tcn_out1)
        cnn_out = F.relu(self.conv1(x))
        cnn_out = F.relu(self.conv2(cnn_out))
        
        # 拼接并应用通道注意力机制
        x = torch.cat((tcn_out2, cnn_out), dim=1)
        x = self.channel_attention(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output


class TCN_Attention_WithAttention(nn.Module):
    def __init__(self, seq_len=1035):
        super(TCN_Attention_WithAttention, self).__init__()
        self.tcn1 = TemporalBlock(1, 64, 3, 1, 1, 1, 0.05)
        self.tcn2 = TemporalBlock(64, 128, 3, 1, 7, 7, 0.05)
        self.attention = AttentionModule(1)
        self.channel_attention = ChannelAttentionModule(129)  # TCN(128) + Attention(1)
        self.fc1 = nn.Linear(129 * seq_len, 128)
        self.fc2 = nn.Linear(128, 2)
        self.lossfunction = nn.CrossEntropyLoss()
    
    def forward(self, x):
        tcn_out1 = self.tcn1(x)
        tcn_out2 = self.tcn2(tcn_out1)
        attention_out = self.attention(x.transpose(1, 2)).transpose(1, 2)
        
        # 拼接并应用通道注意力机制
        x = torch.cat((tcn_out2, attention_out), dim=1)
        x = self.channel_attention(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output


class CNN_Attention_WithAttention(nn.Module):
    def __init__(self, seq_len=1035):
        super(CNN_Attention_WithAttention, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.attention = AttentionModule(1)
        self.channel_attention = ChannelAttentionModule(129)  # CNN(128) + Attention(1)
        self.fc1 = nn.Linear(129 * seq_len, 128)
        self.fc2 = nn.Linear(128, 2)
        self.lossfunction = nn.CrossEntropyLoss()
    
    def forward(self, x):
        cnn_out = F.relu(self.conv1(x))
        cnn_out = F.relu(self.conv2(cnn_out))
        attention_out = self.attention(x.transpose(1, 2)).transpose(1, 2)
        
        # 拼接并应用通道注意力机制
        x = torch.cat((cnn_out, attention_out), dim=1)
        x = self.channel_attention(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output


