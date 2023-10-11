import torch
import torch.nn as nn
from pathlib import Path
from model_v5 import LSTM_V, LSTM_X
import numpy as np
import pdb


def state_predict(states: torch.Tensor, actions: torch.Tensor, timesteps: torch.Tensor, masks: torch.Tensor, device,
                  state_columns, action_columns) -> torch.Tensor:
    """

    输入状态时间序列(序列长度为20), 预测下一个时刻(t=21)的state;
    注意:输入输出均为原始状态

    Args:
        states (torch.tensor): 状态
        actions (torch.tensor): 动作
        timesteps (torch.tensor): 时间步
        mask (torch.tensor): 掩码
        state_columns: state中每一个维度的物理含义
        action_columns: action中每一个维度的物理含义
        device
    return:
        next_states(torch.tensor), 请使用float32类型的tensor

    example usage:

    states = torch.ones(batch_size, seq_length, state_dim)
    actions = torch.ones(batch_size, seq_length, action_dim)
    timesteps = torch.ones(batch_size, seq_length)
    masks = torch.ones(batch_size, seq_length)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state_columns = state_columns = ['pos_boom', 'pos_arm', 'pos_swing', 'vel_boom', 'vel_arm', 'vel_swing', 'state_time', 'action_time']
    action_columns = ['pwm_boom', 'pwm_arm', 'pwm_swing']
    next_state = state_predict(states, actions,timesteps, masks, device, state_columns, action_columns) -> torch.tensor(batch_size, 1, state_dim)

    # """

    input_state_index = [state_columns.index('pos_boom'), state_columns.index('pos_arm'),
                         state_columns.index('pos_swing'),
                         state_columns.index('vel_boom'), state_columns.index('vel_arm'),
                         state_columns.index('vel_swing'),
                         state_columns.index('state_time'), state_columns.index('action_time')]
    # input_a_index = [state_columns.index('pos_boom'), state_columns.index('pos_arm'),
    #                  state_columns.index('pos_swing')]
    input_action_index = [action_columns.index('pwm_boom'), action_columns.index('pwm_arm'),
                          action_columns.index('pwm_swing')]

    input_state = torch.cat([states[:, :, input_state_index], actions[:, :, input_action_index]], dim=2)
    # input_a = states[:, :, input_state_index[0:3]]

    state_time = torch.mean(states[:, :, input_state_index[6]], dim=1).unsqueeze(1)
    action_time = torch.mean(states[:, :, input_state_index[7]], dim=1).unsqueeze(1)
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

    ##  定义归一化数据
    state_max = torch.tensor([1.09432319e+00, 1.26621892e+00, 3.14159265e+00, 6.16101226e-01, 9.44223125e-01,
                              1.06465084e+00, 1.49962902e-01, 1.46156549e-01, 1.00000000e+03, 1.00000000e+03,
                              1.00000000e+03]).unsqueeze(0).to(device)
    state_min = torch.tensor([-9.90472605e-01, -2.16282946e+00, -3.14131880e+00, -5.23598776e-01, -9.02160690e+00,
                              -1.21125850e+00, 0.00000000e+00, 0.00000000e+00, -1.00000000e+03,
                              -1.00000000e+03, -1.00000000e+03]).unsqueeze(0).to(device)
    # lnn_max = torch.tensor([1.09432319e+00, 1.26621892e+00, 3.14159265e+00, 6.16101226e-01, 9.44223125e-01,
    #                         1.06465084e+00, 1.49962902e-01, 1.46156549e-01]).unsqueeze(0).to(device)
    # lnn_min = torch.tensor([-9.90472605e-01, -2.16282946e+00, -3.14131880e+00, -5.23598776e-01, -9.02160690e+00,
    #                         -1.21125850e+00, 0.00000000e+00, 0.00000000e+00]).unsqueeze(0).to(device)
    # A_max = torch.tensor([8.32166800e+00, 8.25934400e+00, 8.50648400e+00]).unsqueeze(0).to(device)
    # A_min = torch.tensor([-8.28961600e+00, -8.13061700e+00, -6.14461540e-01]).unsqueeze(0).to(device)

    x_max = torch.tensor([1.09432319e+00, 1.26621892e+00, 3.14159265e+00]).unsqueeze(0).to(device)
    x_min = torch.tensor([-9.90472605e-01, -2.16282946e+00, -3.14131880e+00]).unsqueeze(0).to(device)

    v_max = torch.tensor([6.16101226e-01, 9.44223125e-01, 1.06465084e+00]).unsqueeze(0).to(device)
    v_min = torch.tensor([-5.23598776e-01, -9.02160690e+00, -1.21125850e+00]).unsqueeze(0).to(device)

    lstm_in_norm = ((input_state - state_min) / (state_max - state_min)).to(device)

    ##  载入模型

    # net_A = FC_A(lnn_in=3, lnn_hid1=64, lnn_hid2=512, lnn_hid3=1024, lnn_hid4=512, lnn_hid5=64, lnn_out=3).cuda()  # A2a
    # net_a = FC_a(lnn_in=3, lnn_hid1=64, lnn_hid2=512, lnn_hid3=1024, lnn_hid4=512, lnn_hid5=64, lnn_out=3).cuda()  # a2A
    net_v = LSTM_V(lstm_in=11, lstm_hid=512, lstm_layer=12, lstm_out=3, lnn_in=6,
                   lnn_hid1=128, lnn_hid2=512, lnn_hid3=2048, lnn_hid4=512, lnn_hid5=128, lnn_hid6=32,
                   lnn_out=3).cuda()  # 定义模型
    net_x = LSTM_X(lstm_in=11, lstm_hid=512, lstm_layer=16, lstm_out=3, lnn_in=6,
                   lnn_hid1=128, lnn_hid2=512, lnn_hid3=2048, lnn_hid4=512, lnn_hid5=128, lnn_hid6=32,
                   lnn_out=3).cuda()  # 定义模型

    world_model_path = str(Path(__file__).parent.absolute() / 'net_final_v5.pth')
    model_all = torch.load(world_model_path)

    net_v.load_state_dict(model_all['model_v'])
    net_v.to(device)
    net_v.eval()

    net_x.load_state_dict(model_all['model_x'])
    net_x.to(device)
    net_x.eval()

    # 开始计算
    # cal_A_norm = net_a(a_in_norm[:, -1, :])  # net_a为a输入， a2A
    # cal_A = cal_A_norm.detach() * (A_max - A_min) + A_min
    # pdb.set_trace()
    # cal_next_A_norm = net_x(lstm_in_norm, cal_A_norm)
    # cal_next_A = cal_next_A_norm.detach() * (A_max - A_min) + A_min

    # (batch_size, seq_length, state_dim)
    cal_next_v_norm = net_v(lstm_in_norm[:, -128:, :])
    cal_next_x_norm = net_x(lstm_in_norm[:, -128:, :])
    #
    # cal_next_v_norm = net_v(lstm_in_norm)
    # cal_next_x_norm = net_x(lstm_in_norm)

    cal_next_v = cal_next_v_norm.detach() * (v_max - v_min) + v_min
    cal_next_x = cal_next_x_norm.detach() * (x_max - x_min) + x_min

    next_states = torch.cat([cal_next_x, cal_next_v, state_time, action_time], dim=1).unsqueeze(1)

    # next_states = torch.cat([cal_next_x_out[:, :, 0:3], cal_next_v_out, cal_next_x_out[:, :, -2:]], dim=2)
    # next_states = torch.cat([cal_next_x_out[:, :, 0:3], cal_next_v_out, cal_next_x_out[:, :, -2:]], dim=2)
    return next_states
