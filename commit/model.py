import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, lstm_output_size, lnn_input_size, lnn_hidden_size_1, lnn_hidden_size_2,
                 lnn_hidden_size_3, lnn_hidden_size_4, lnn_output_size):  # (6, 3, 1)
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # rnn
        self.fc = nn.Linear(hidden_size, lstm_output_size)
        self.fc_LNN_in = nn.Linear(lnn_input_size, lnn_hidden_size_1)
        self.fc_relu = nn.ReLU()
        self.fc_sigmoid = nn.Sigmoid()
        self.fc_LNN_mid_1 = nn.Linear(lnn_hidden_size_1, lnn_hidden_size_2)
        self.fc_LNN_mid_2 = nn.Linear(lnn_hidden_size_2, lnn_hidden_size_3)
        self.fc_LNN_mid_3 = nn.Linear(lnn_hidden_size_3, lnn_hidden_size_4)
        self.fc_LNN_out = nn.Linear(lnn_hidden_size_4, lnn_output_size)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, u, mid_x):
        lstm_output_ = self.rnn(u)
                # pdb.set_trace()
        lstm_output = lstm_output_[1][0][0, :]  # 计算Lstm的h
        # lstm_output = self.dropout(lstm_output)
        cal_next_v = self.fc(lstm_output)  # 计算Lstm的输出
        lnn_input = torch.cat([cal_next_v, mid_x], dim=1)  # new_input(50, 8)
        lnn_hidden_1 = self.fc_relu(self.fc_LNN_in(lnn_input))
        lnn_hidden_2 = self.fc_relu(self.fc_LNN_mid_1(lnn_hidden_1))
        lnn_hidden_3 = self.fc_relu(self.fc_LNN_mid_2(lnn_hidden_2))
        lnn_hidden_4 = self.fc_relu(self.fc_LNN_mid_3(lnn_hidden_3))
        cal_next_x = self.fc_sigmoid(self.fc_LNN_out(lnn_hidden_4))
        return cal_next_v, cal_next_x
