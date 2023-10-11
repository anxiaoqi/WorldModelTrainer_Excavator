import torch
import torch.nn as nn
import pdb


class LSTM_V(nn.Module):
    def __init__(self, lstm_in, lstm_hid, lstm_layer, lstm_out,
                 lnn_in, lnn_hid1, lnn_hid2, lnn_hid3, lnn_hid4, lnn_hid5, lnn_hid6, lnn_out):  # (6, 3, 1)
        super(LSTM_V, self).__init__()
        self.rnn = nn.LSTM(lstm_in, lstm_hid, lstm_layer, batch_first=True)  # rnn
        self.fc = nn.Linear(lstm_hid, lstm_out)
        self.fc_in = nn.Linear(lstm_out+lnn_in, lnn_hid1)
        self.fc_hid1 = nn.Linear(lnn_hid1, lnn_hid2)
        self.fc_hid2 = nn.Linear(lnn_hid2, lnn_hid3)
        self.fc_hid3 = nn.Linear(lnn_hid3, lnn_hid4)
        self.fc_hid4 = nn.Linear(lnn_hid4, lnn_hid5)
        self.fc_hid5 = nn.Linear(lnn_hid5, lnn_hid6)
        self.fc_out = nn.Linear(lnn_hid6, lnn_out)

        self.fc_relu = nn.ReLU()
        self.fc_sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, u):
        lstm_h = self.rnn(u)
                # pdb.set_trace()
        lstm_h_end = lstm_h[0][:, -1, :]  # 计算Lstm的h, [layer, batch, hid], output:[batch, sequence, hid]
        # lstm_output = self.dropout(lstm_output)
        lstm_out = self.fc(lstm_h_end)  # 计算Lstm的输出


        lnn_in = torch.cat([lstm_out, u[:, -1, 3:6], u[:, -1, -3:]], dim=1)  # new_input(50, 8)
        lnn_hid1 = self.fc_relu(self.fc_in(lnn_in))
        lnn_hid2 = self.fc_relu(self.fc_hid1(lnn_hid1))
        lnn_hid3 = self.fc_relu(self.fc_hid2(lnn_hid2))
        lnn_hid4 = self.fc_relu(self.fc_hid3(lnn_hid3))
        lnn_hid5 = self.fc_relu(self.fc_hid4(lnn_hid4))
        # lnn_hid5 = self.dropout(lnn_hid5)
        lnn_hid6 = self.fc_relu(self.fc_hid5(lnn_hid5))
        cal_next_v = self.fc_sigmoid(self.fc_out(lnn_hid6))
        return cal_next_v
