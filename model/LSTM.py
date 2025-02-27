import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # 初始化隐藏状态h0, c0为全0向量

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)


        # 将输入x和隐藏状态(h0, c0)传入LSTM网络
        out, _ = self.lstm( x,(h0, c0))
        # 取最后一个时间步的输出作为LSTM网络的输出
        # out = self.linear(out[:, -1, :])
        out = self.linear(out)
        return out
