import data_input
import torch
import torch.nn as nn
import glob
import pandas as pd
import numpy as np
import random
import pdb
from model import LSTM
from predict import state_predict

# 确认Pytorch
# print(torch.__version__)
# print(torch.cuda.is_available())

#  准备数据
# number        state_time  	action_time     	pos_boom
# pos_arm	    pos_swing	    vel_boom	        vel_arm
# vel_swing	    next_pos_boom	next_pos_arm	    next_pos_swing
# next_vel_boom	next_vel_arm	next_vel_swing	    pwm_boom
# pwm_arm	    pwm_swing
data_max = np.array([2.34440000e+04, 1.49962902e-01, 1.21906042e-01, 6.49962359e-01,
                     1.25976120e+00, 3.14159265e+00, 6.03883921e-01, 7.33038286e-01,
                     1.00880031e+00, 6.49962359e-01, 1.25976120e+00, 3.14159265e+00,
                     6.03883921e-01, 7.33038286e-01, 1.00880031e+00, 1.00000000e+03,
                     1.00000000e+03, 1.00000000e+03])

data_min = np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -8.60271043e-01,
                     -8.82630456e-01, -3.13984732e+00, -4.55530935e-01, -1.15191731e+00,
                     -8.93608577e-01, -8.60271043e-01, -8.82630456e-01, -3.13984732e+00,
                     -4.55530935e-01, -1.15191731e+00, -8.93608577e-01, -1.00000000e+03,
                     -1.00000000e+03, -1.00000000e+03])

data_list = []
data_list_norm = []
path_folder = "D:/4_Doctor_work/2_NetEase_Fuxi_Algorithm_Competition/1_data_set/release_data/filtered_data_new/train/*"
for folder_abs in glob.glob(path_folder):
    # print(folder_abs)
    path_file = folder_abs + "/*"
    for file_abs in glob.glob(path_file):
        # print(file_abs)
        contend = pd.read_csv(file_abs)
        contend_norm = (contend - data_min) / (data_max - data_min)
        # data.info()
        data_list.append(contend)
        data_list_norm.append(contend_norm)

# 加载模型
# input_size, hidden_size, num_layers, lstm_output_size, nn_input_size(3+5), nn_output_size
net = LSTM(input_size=8, hidden_size=40, num_layers=6, lstm_output_size=3, lnn_input_size=8, lnn_hidden_size_1=32,
           lnn_hidden_size_2=64, lnn_output_size=5).cuda()  # 定义模型
loss = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)  # 定义优化器
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
length = 20  # 序列长度
batch = 128  # batch尺寸

# 开始训练
print("Training......")
for e in range(100000):
    # 加载数据
    page = random.randint(0, len(data_list_norm) - 1)
    # print(page)
    data = data_list_norm[page]
    if len(data) < length + batch:
        continue
    u_prep, next_v_prep, mid_x_prep, next_x_prep, start = data_input.extract_data(data, length, batch)
    input_u = torch.tensor(u_prep, dtype=torch.float32).cuda()
    input_next_v = torch.tensor(next_v_prep, dtype=torch.float32).cuda()
    input_mid_x = torch.tensor(mid_x_prep, dtype=torch.float32).cuda()
    input_next_x = torch.tensor(next_x_prep, dtype=torch.float32).cuda()

    cal_next_v, cal_next_x = net(input_u, input_mid_x)  # 向前传播  (50, 1 , 3)
    # pdb.set_trace()
    Loss1 = loss(cal_next_v, input_next_v)  # 计算损失 #()
    Loss2 = loss(cal_next_x, input_next_x)
    Loss = Loss1 + Loss2
    optimizer.zero_grad()  # 梯度清零
    Loss.backward()  # 反向传播
    optimizer.step()  # 梯度更新
    # scheduler.step()

    if e % 10000 == 0:
        torch.save(net, "net" + str(e) + ".pth")
        print('Epoch: {:4}, Loss: {:.5f}'.format(e, Loss.item()))

torch.save(net, "net_final.pth")
# 测试数据
data_test_list = []
path_folder = "D:/4_Doctor_work/2_NetEase_Fuxi_Algorithm_Competition/1_data_set/release_data/filtered_data_new/test/*"
for folder_abs in glob.glob(path_folder):
    # print(folder_abs)
    path_file = folder_abs + "/*"
    for file_abs in glob.glob(path_file):
        # print(file_abs)
        contend = pd.read_csv(file_abs)
        contend_norm = (contend - data_min) / (data_max - data_min)
        # data.info()
        data_test_list.append(contend)

length_test = 20
batch_test = 50
# page_test = random.randint(0, len(data_list_norm) - 1)
page_test = 2

States = data_test_list[page_test].loc[:,
         ['pos_boom', 'pos_arm', 'pos_swing', 'vel_boom', 'vel_arm', 'vel_swing', 'state_time', 'action_time']]
Actions = data_test_list[page_test].loc[:,
          ['pwm_boom', 'pwm_arm', 'pwm_swing']]

States_input = torch.tensor(np.array(States[50:70]), dtype=torch.float32).unsqueeze(0)
Actions_input = torch.tensor(np.array(Actions[50:70]), dtype=torch.float32).unsqueeze(0)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
state_columns = ['pos_boom', 'pos_arm', 'pos_swing', 'vel_boom', 'vel_arm', 'vel_swing', 'state_time', 'action_time']
action_columns = ['pwm_boom', 'pwm_arm', 'pwm_swing']
next_states = state_predict(States_input, Actions_input, [], [], device,
                            state_columns, action_columns)

pdb.set_trace()
# print(cal_next_v_out)
# print(cal_next_x_out)
# pdb.set_trace()
