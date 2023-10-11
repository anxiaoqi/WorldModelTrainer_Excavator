import torch
import torch.nn as nn
import glob
import pandas as pd
import numpy as np
import random
import pdb
from model_x import LSTM_X


# 确认Pytorch
# print(torch.__version__)
# print(torch.cuda.is_available())

def extract_data(u, next_x, length, batch):  # define method to extract X and Y
    input_u = []
    output_next_x = []

    start = random.randint(0, len(u) - batch - length)
    # for i in range(len(data) - length + 1):
    for i in range(start, start + batch):
        input_u.append(u[i:i + length])
        output_next_x.append(next_x[i + length - 1:i + length])

    # pdb.set_trace()
    u_prep = np.array(input_u)
    next_x_prep = np.array(output_next_x)
    next_x_prep = next_x_prep[:, -1, :]

    return u_prep, next_x_prep, start


# def wgn(x, snr):
#     Ps = np.sum(abs(x), axis=0) / len(x)
#     Pn = Ps / (10 ** ((snr / 10)))
#     # np.expand_dims(Pn, axis=0)
#     noise = np.matmul(np.random.randn(len(x), 1), np.expand_dims(Pn, axis=0))
#     # noise = np.random.randn(len(x), 1) * np.transpose(np.sqrt(Pn))
#     signal_add_noise = x + noise
#     return signal_add_noise


#  准备数据


# data_max = np.array([3.00230000e+04, 1.49962902e-01, 1.46156549e-01, 1.09432319e+00,
#                      1.26621892e+00, 3.14159265e+00, 6.16101226e-01, 9.44223125e-01,
#                      1.06465084e+00, 1.09432319e+00, 1.26621892e+00, 3.14159265e+00,
#                      6.16101226e-01, 9.44223125e-01, 1.06465084e+00, 1.00000000e+03,
#                      1.00000000e+03, 1.00000000e+03, 8.32166800e+00, 8.25934400e+00,
#       	             8.50648400e+00, 8.32187900e+00, 8.25934400e+00, 8.50648400e+00])
#
# data_min = np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -9.90472605e-01,
#                      -2.16282946e+00, -3.14131880e+00, -5.23598776e-01, -9.02160690e+00,
#                      -1.21125850e+00, -9.90298072e-01, -2.16282946e+00, -3.14131880e+00,
#                      -5.23598776e-01, -9.02160690e+00, -1.21125850e+00, -1.00000000e+03,
#                      -1.00000000e+03, -1.00000000e+03, -8.28961600e+00, -8.13061700e+00,
#                      -6.14461540e-01, -8.28961600e+00, -8.13061700e+00, -6.14461540e-01])
# number        state_time  	action_time     	pos_boom
# pos_arm	    pos_swing	    vel_boom	        vel_arm
# vel_swing	    next_pos_boom	next_pos_arm	    next_pos_swing
# next_vel_boom	next_vel_arm	next_vel_swing	    pwm_boom
# pwm_arm	    pwm_swing       x                   y
# z             next_x          next_y              next_z

data_u_max = np.array(
    [1.09432319e+00, 1.26621892e+00, 3.14159265e+00, 6.16101226e-01, 9.44223125e-01, 1.06465084e+00,
     1.49962902e-01, 1.46156549e-01, 1.00000000e+03, 1.00000000e+03, 1.00000000e+03])
data_u_min = np.array(
    [-9.90472605e-01, -2.16282946e+00, -3.14131880e+00, -5.23598776e-01, -9.02160690e+00, -1.21125850e+00,
     0.00000000e+00, 0.00000000e+00, -1.00000000e+03, -1.00000000e+03, -1.00000000e+03])
x_max = np.array([1.09432319e+00, 1.26621892e+00, 3.14159265e+00])
x_min = np.array([-9.90472605e-01, -2.16282946e+00, -3.14131880e+00])
# data_x_max = np.array([ 8.32166800e+00, 8.25934400e+00, 8.50648400e+00])
# data_x_min = np.array([-8.28961600e+00, -8.13061700e+00, -6.14461540e-01])
# data_next_max = np.array([8.32187900e+00, 8.25934400e+00, 8.50648400e+00])
# data_next_min = np.array([-8.28961600e+00, -8.13061700e+00, -6.14461540e-01])

u_list = []
mid_list = []
next_x_list = []
u_x_list = []

# path_folder = "/home/ubuntu/4T/mdx/2_setdata/train/*"
# path_folder = "D:/4_Doctor_work/2_NetEase_Fuxi_Algorithm_Competition/1_data_set/release_data/filtered_data_new/train/*"
# path_folder = "D:/4_Doctor_work/2_NetEase_Fuxi_Algorithm_Competition/2_第二阶段/1_dataset/transfer/test0/*"
path_folder = "/media/seu404/4T/mdx/2_Netease_setup2/0_dataset/transfer/train/*"

for folder_abs in glob.glob(path_folder):
    # print(folder_abs)
    path_file = folder_abs + "/*"
    for file_abs in glob.glob(path_file):
        # print(file_abs)
        contend = pd.read_csv(file_abs)
        contend_u = contend.loc[:,
                    ['pos_boom', 'pos_arm', 'pos_swing', 'vel_boom', 'vel_arm', 'vel_swing', 'state_time',
                     'action_time', 'pwm_boom', 'pwm_arm', 'pwm_swing']][0:-1]
        # contend_u_x = contend.loc[:, ['x', 'y', 'z']][0:-1]
        # contend_next_x = contend.loc[:, ['next_x', 'next_y', 'next_z']][0:-1]
        contend_next_x = contend.loc[:, ['next_pos_boom', 'next_pos_arm', 'next_pos_swing']][0:-1]

        # contend_u_noise = wgn(contend_u, 16)
        contend_u_norm = (contend_u - data_u_min) / (data_u_max - data_u_min)
        contend_next_x_norm = (contend_next_x - x_min) / (x_max - x_min)
        # contend_u_x_norm = (contend_u_x - data_x_min) / (data_x_max - data_x_min)
        # data.info()
        # data_list.append(contend)

        u_list.append(contend_u_norm)
        # u_x_list.append(contend_u_x_norm)
        next_x_list.append(contend_next_x_norm)

# 加载模型

# net_x = LSTM_X(input_size=11, hidden_size=32, num_layers=8, lstm_output_size=8, lnn_input_size=16, lnn_hidden_size_1=32,
#                lnn_hidden_size_2=64, lnn_hidden_size_3=128, lnn_hidden_size_4=64, lnn_hidden_size_5=32,
#                lnn_hidden_size_6=16, lnn_output_size=3).cuda()  # 定义模型

# net_x = LSTM_X(lstm_in=11, lstm_hid=256, lstm_layer=16, lstm_out=3, lnn_in=6,
#                lnn_hid1=64, lnn_hid2=256, lnn_hid3=1024, lnn_hid4=256, lnn_hid5=64, lnn_hid6=16,
#                lnn_out=3).cuda()  # 定义模型

net_x = LSTM_X(lstm_in=11, lstm_hid=512, lstm_layer=16, lstm_out=3, lnn_in=6,
               lnn_hid1=128, lnn_hid2=512, lnn_hid3=2048, lnn_hid4=512, lnn_hid5=128, lnn_hid6=32,
               lnn_out=3).cuda()  # 定义模型

# net_x = LSTM_X(lstm_in=11, lstm_hid=32, lstm_layer=8, lstm_out=8, lnn_in=8,
#                lnn_hid1=16, lnn_hid2=32, lnn_hid3=32, lnn_hid4=32, lnn_hid5=32, lnn_hid6=16, lnn_out=3).cuda()
loss = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.SGD(net_x.parameters(), lr=0.1)  # 定义优化器
# optimizer = torch.optim.Adam(net_x.parameters(), lr=1e-3, eps=1e-8)  # 定义优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.1)
length = 128  # 序列长度
batch = 256  # batch尺寸

# 开始训练
last_Loss = 100
print("Training......")
for e in range(1500001):
    # 加载数据
    page = random.randint(0, len(u_list) - 1)
    # print(page)
    in_u = u_list[page]
    in_next_x = next_x_list[page]

    if len(in_u) < length + batch:
        continue
    u_prep, next_x_prep, start = extract_data(in_u, in_next_x, length, batch)

    input_u = torch.tensor(u_prep, dtype=torch.float32).cuda()
    input_next_x = torch.tensor(next_x_prep, dtype=torch.float32).cuda()
    # pdb.set_trace()
    cal_next_x = net_x(input_u)  # 向前传播  (50, 1 , 3)
    # pdb.set_trace()
    Loss = 10 * loss(cal_next_x, input_next_x)  # 计算损失 #()

    if Loss < last_Loss:
        checkpoint = {'epoch': e,
                      'best_loss': Loss,
                      'model': net_x.state_dict(),
                      'optimizer': optimizer.state_dict()
                      }
        torch.save(checkpoint, 'net_final.pth')
        # torch.save(net, "net_final.pth")
        print(Loss, e)
        last_Loss = Loss

    if Loss < 1e-10:
        checkpoint = {'epoch': e,
                      'best_loss': Loss,
                      'model': net_x.state_dict(),
                      'optimizer': optimizer.state_dict()
                      }
        torch.save(checkpoint, f='net_' + str(Loss) + '.pth')

    optimizer.zero_grad()  # 梯度清零
    Loss.backward()  # 反向传播
    optimizer.step()  # 梯度更新
    scheduler.step()

    if e % 5000 == 0:
        # torch.save(net_x, "net" + str(e) + ".pth")
        print('Epoch: {:4}, Loss: {:.5f}'.format(e, Loss.item()))
