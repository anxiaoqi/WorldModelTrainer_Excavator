import glob
import pandas as pd
import numpy as np
import random
import pdb
import torch
import torch.nn as nn
import torch.optim as optim


def extract_data(data, length, batch):  # define method to extract X and Y
    input_u = []
    output_next_v = []
    input_mid_x = []
    output_next_x = []

    # LSTM输入 8个输入
    # state_columns = state_columns = ['pos_boom', 'pos_arm', 'pos_swing', 'vel_boom', 'vel_arm', 'vel_swing',
    #                                  'state_time', 'action_time']
    # action_columns = ['pwm_boom', 'pwm_arm', 'pwm_swing']
    u = data.loc[:,
        ['vel_boom', 'vel_arm', 'vel_swing', 'state_time', 'action_time',
         'pwm_boom', 'pwm_arm', 'pwm_swing']]
    # LSTM输出 3个速度
    next_v = data.loc[:,
             ['next_vel_boom', 'next_vel_arm', 'next_vel_swing']]
    # LNN输入 5个输入 + 3个速度
    mid_x = data.loc[:,
            ['pos_boom', 'pos_arm', 'pos_swing', 'state_time', 'action_time']]
    # LNN输出 3个位置+2个时间
    next_x = data.loc[:,
             ['next_pos_boom', 'next_pos_arm', 'next_pos_swing', 'state_time', 'action_time']]

    start = random.randint(0, len(data) - batch - length)
    # for i in range(len(data) - length + 1):
    for i in range(start, start + batch):
        # print(i) u, next_v, mid_x, next_x
        input_u.append(u[i:i + length])
        output_next_v.append(next_v[i + length - 1:i + length])
        input_mid_x.append(mid_x[i + length - 1:i + length])
        output_next_x.append(next_x[i + length - 1:i + length])
    # pdb.set_trace()
    u_prep = np.array(input_u)
    next_v_prep = np.array(output_next_v)
    next_v_prep = next_v_prep[:, -1, :]
    mid_x_prep = np.array(input_mid_x)
    mid_x_prep = mid_x_prep[:, -1, :]
    next_x_prep = np.array(output_next_x)
    next_x_prep = next_x_prep[:, -1, :]
    return u_prep, next_v_prep, mid_x_prep, next_x_prep, start


def anti_norm_x(data, data_max, data_min):
    data = np.array(data)
    data_x_max = data_max[9:12]
    data_x_min = data_min[9:12]
    data_out = data * (data_x_max - data_x_min) + data_x_min
    return data_out


def anti_norm_v(data, data_max, data_min):
    data = np.array(data)
    data_v_max = data_max[12:15]
    data_v_min = data_min[12:15]
    data_out = data * (data_v_max - data_v_min) + data_v_min
    return data_out
