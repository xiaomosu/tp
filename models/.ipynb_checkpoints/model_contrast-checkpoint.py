import torch.nn as nn
#对比实验1+消融实验----------------------------------------------------------------------------------------------------------
class tcn(nn.Module):
    def __init__(self):
        super(tcn, self).__init__()
        self.conv_1 = nn.Conv1d(1, 64, kernel_size=2, dilation=1, padding=((2-1) * 1))
        self.conv_2 = nn.Conv1d(64, 32, kernel_size=4, dilation=2, padding=((4-1) * 2))
        self.conv_3 = nn.Conv1d(32, 16, kernel_size=8, dilation=4, padding=((8-1) * 4))
        self.conv_4 = nn.Conv1d(16, 8, kernel_size=16, dilation=8, padding=((16-1) * 8))
        self.conv_5 = nn.Conv1d(8, 4, kernel_size=32, dilation=16, padding=((32-1) * 16))
        self.dense_1 = nn.Linear(1035*4, 128)
        self.dense_2 = nn.Linear(128, 2)
        self.lossfunction = nn.CrossEntropyLoss()
        # self.reduce_dim = nn.Linear(64, 2)  # 使 x2 的维度从 64 降到 2
    def forward(self, x):
        x = self.conv_1(x)
        x = x[:, :, :-self.conv_1.padding[0]]
        x = F.relu(x)
        x = F.dropout(x, 0.05)
        x = self.conv_2(x)
        x = x[:, :, :-self.conv_2.padding[0]]
        x = F.relu(x)
        x = F.dropout(x, 0.05)
        x = self.conv_3(x)
        x = x[:, :, :-self.conv_3.padding[0]]
        x = F.relu(x)
        x = F.dropout(x, 0.05)
        x = self.conv_4(x)
        x = x[:, :, :-self.conv_4.padding[0]]
        x = F.relu(x)
        x = F.dropout(x, 0.05)
        x = self.conv_5(x)
        x = x[:, :, :-self.conv_5.padding[0]]
        x = F.relu(x)
        x = F.dropout(x, 0.05)
        x = x.view(-1, 1035*4)
        x = F.relu(self.dense_1(x))
        x = self.dense_2(x)
        return x
#对比实验2----------------------------------------------------------------------------------------------------------
class fedlstmgru(nn.Module):
    def __init__(self, input_size=1035, hidden_size=64, num_layers=1):
        super(fedlstmgru, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        # Dense layers
        self.dense_1 = nn.Linear(hidden_size, 128)
        self.dense_2 = nn.Linear(128, 2)

        # Loss function
        self.lossfunction = nn.CrossEntropyLoss()

    def forward(self, x):
        # Initial hidden and cell states for LSTM
        h0_lstm = torch.zeros(1, x.size(0), 64).to(x.device)
        c0_lstm = torch.zeros(1, x.size(0), 64).to(x.device)

        # LSTM forward pass
        out_lstm, _ = self.lstm(x, (h0_lstm, c0_lstm))

        # Initial hidden state for GRU
        h0_gru = torch.zeros(1, out_lstm.size(0), 64).to(out_lstm.device)

        # GRU forward pass
        out_gru, _ = self.gru(out_lstm, h0_gru)

        # Use the output of the last time step from the GRU
        x = out_gru[:, -1, :]

        # Fully connected layers
        x = F.relu(self.dense_1(x))
        x = self.dense_2(x)

        return x

#对比实验三、lr=0.0001---------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, num_heads, num_layers, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        x = self.embedding(x)
        # Now Transformer expects input in the shape: [batch_size, sequence_length, d_model]
        x = self.transformer_encoder(x)
        return x

class SVDD(nn.Module):
    def __init__(self, input_dim, c=1.0):
        super(SVDD, self).__init__()
        self.center = nn.Parameter(torch.randn(input_dim))  # 初始化中心
        self.radius = nn.Parameter(torch.tensor(0.0))  # 初始化半径
        self.c = c  # 正则化参数

    def forward(self, x):
        # 计算距离平方: ||x - center||^2
        dist = torch.sum((x - self.center) ** 2, dim=1)
        # 计算SVDD损失，最大化球体内的点
        loss = torch.mean(F.relu(dist - self.radius**2)) + self.c * torch.sum(self.radius)
        return dist, loss


# TransformerVAE Model
class TransformerVAE_svdd(nn.Module):
    def __init__(self, input_size=1035, d_model=64, num_heads=4, num_layers=2, dim_feedforward=128, latent_dim=32):
        super(TransformerVAE_svdd, self).__init__()
        self.encoder = TransformerEncoder(input_size, d_model, num_heads, num_layers, dim_feedforward)
        self.fc_mu = nn.Linear(d_model, latent_dim)  # Mean of latent space
        self.fc_logvar = nn.Linear(d_model, latent_dim)  # Log variance of latent space
        self.fc_decode = nn.Linear(latent_dim, d_model)
        self.decoder = nn.Linear(d_model, 128)  # Modify to 128 units for classification head
        self.classifier = nn.Linear(128, 2)  # 2 classes for classification
        # 定义 SVDD 模型，使用编码器的输出作为输入
        self.svdd = SVDD(input_dim=latent_dim)

        # Loss function
        self.lossfunction = nn.CrossEntropyLoss()

    def encode(self, x):
        h = self.encoder(x)
        # Use the last time step from Transformer output
        mu = self.fc_mu(h[:, -1, :])  # [batch_size, latent_dim]
        logvar = self.fc_logvar(h[:, -1, :])  # [batch_size, latent_dim]
        logvar = torch.clamp(logvar, min=-10, max=10)  # Prevent logvar from becoming too extreme
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-6)  # Clamp std to avoid NaN from very small values
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc_decode(z))
        h = torch.relu(self.decoder(h))  # Decoder output for classification
        return self.classifier(h)  # Output for classification

    def forward(self, x):
        # Encode input
        mu, logvar = self.encode(x)
        # Reparameterize latent space
        z = self.reparameterize(mu, logvar)
        # Decode latent vector into class scores
        output = self.decode(z)
        return output  # Only return classification output

