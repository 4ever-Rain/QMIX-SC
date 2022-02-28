import torch.nn as nn
import torch.nn.functional as f
import torch


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class TRANSFORER(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(TRANSFORER, self).__init__()
        self.args = args
        # self.seq_length = 10
        # self.seq = []
        # for _ in range(self.seq_length):
        #     self.seq.append(torch.rand_like(input_shape))
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)

        self.pos_emb = nn.Parameter(torch.rand(1, 1, self.args.rnn_hidden_dim))
        # self.pos_emb = nn.Parameter(torch.rand(1, self.args.rnn_hidden_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(args.rnn_hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs):
        # if len(self.seq) <= self.seq_length:
        #     self.seq.append(obs)
        # 判断
        # 组成队列
        # （batch，seq，embin）
        # torch.cat / stack
        # transformer_encoder要求输入为(N, seq， embeding)
        # 第一种解决方案，直接加一个维度加为seq(1)
        # torch.autograd.set_detect_anomaly(True)
        x = f.relu(self.fc1(obs))
        x = x.unsqueeze(dim=1)
        x = x + self.pos_emb
        h = self.transformer_encoder(x)
        h = h.squeeze(dim=1)
        q = self.fc2(h)
        return q
