import torch
# from model import LSTM
from pathlib import Path

# 函数格式
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

    # u = data.loc[:,
    #     ['vel_boom', 'vel_arm', 'vel_swing', 'state_time', 'action_time',
    #      'pwm_boom', 'pwm_arm', 'pwm_swing']]
    # # LNN输入 5个输入 + 3个速度
    # mid_x = data.loc[:,
    #         ['pos_boom', 'pos_arm', 'pos_swing', 'state_time', 'action_time']]
    lstm_input_index_1 = [state_columns.index('vel_boom'), state_columns.index('vel_arm'),
                          state_columns.index('vel_swing'),
                          state_columns.index('state_time'), state_columns.index('action_time')]
    lstm_input_index_2 = [action_columns.index('pwm_boom'), action_columns.index('pwm_arm'),
                          action_columns.index('pwm_swing')]
    lnn_input_index = [state_columns.index('pos_boom'), state_columns.index('pos_arm'),
                       state_columns.index('pos_swing'),
                       state_columns.index('state_time'), state_columns.index('action_time')]

    state_lstm_input = torch.cat([states[:, :, lstm_input_index_1], actions[:, :, lstm_input_index_2]], dim=2)
    state_lnn_input = states[:, :, lnn_input_index]

    lstm_input_max = torch.tensor(
        [6.03883921e-01, 7.33038286e-01, 1.00880031e+00, 1.49962902e-01, 1.21906042e-01, 1.00000000e+03, 1.00000000e+03,
         1.00000000e+03]).unsqueeze(0).unsqueeze(0).to(device)
    lstm_input_min = torch.tensor(
        [-4.55530935e-01, -1.15191731e+00, -8.93608577e-01, 0.00000000e+00, 0.00000000e+00, -1.00000000e+03,
         -1.00000000e+03, -1.00000000e+03]).unsqueeze(0).unsqueeze(0).to(device)
    lstm_output_max = torch.tensor([6.03883921e-01, 7.33038286e-01, 1.00880031e+00]).unsqueeze(0).unsqueeze(0).to(device)
    lstm_output_min = torch.tensor([-4.55530935e-01, -1.15191731e+00, -8.93608577e-01]).unsqueeze(0).unsqueeze(0).to(device)
    lnn_input_max = torch.tensor(
        [6.49962359e-01, 1.25976120e+00, 3.14159265e+00, 1.49962902e-01, 1.21906042e-01]).unsqueeze(0).unsqueeze(0).to(device)
    lnn_input_min = torch.tensor(
        [-8.60271043e-01, -8.82630456e-01, -3.13984732e+00, 0.00000000e+00, 0.00000000e+00]).unsqueeze(0).unsqueeze(0).to(device)
    lnn_output_max = torch.tensor([6.49962359e-01, 1.25976120e+00, 3.14159265e+00, 1.49962902e-01, 1.21906042e-01]).unsqueeze(0).unsqueeze(0).to(device)
    lnn_output_min = torch.tensor([-8.60271043e-01, -8.82630456e-01, -3.13984732e+00, 0.00000000e+00, 0.00000000e+00]).unsqueeze(0).unsqueeze(0).to(device)

    state_lstm_input = ((state_lstm_input - lstm_input_min) / (lstm_input_max - lstm_input_min)).to(device)
    state_lnn_input = ((state_lnn_input - lnn_input_min) / (lnn_input_max - lnn_input_min))[:, -1, :].to(device)

    # net = LSTM(input_size=8, hidden_size=20, num_layers=4, lstm_output_size=3, lnn_input_size=8, lnn_hidden_size=20,
    #            lnn_output_size=3).to(device)  # 定义模型
    # net.load_state_dict(torch.load('net10000.pth'))
    world_model_path = str(Path(__file__).parent.absolute() / 'net_final.pth')
    net = torch.load(world_model_path).to(device)
    # net.eval()
    # import pdb
    # pdb.set_trace()
    cal_next_v, cal_next_x = net(state_lstm_input, state_lnn_input)
    cal_next_v_out = cal_next_v.detach() * (lstm_output_max - lstm_output_min) + lstm_output_min
    cal_next_x_out = cal_next_x.detach() * (lnn_output_max - lnn_output_min) + lnn_output_min
    next_states = torch.cat([cal_next_x_out[:, :, 0:3], cal_next_v_out, cal_next_x_out[:, :, -2:]], dim=2)
    return next_states.permute(1, 0, 2)
