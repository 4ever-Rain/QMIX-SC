import torch.nn as nn
import torch.nn.functional as f


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


class BCRNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(BCRNN, self).__init__()
        self.args = args

        self.q_fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.q_rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.q_fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.i_fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.i_rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.i_fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state_q, hidden_state_i):
        x = f.relu(self.q_fc1(obs))
        q_h_in = hidden_state_q.reshape(-1, self.args.rnn_hidden_dim)
        q_h = self.q_rnn(x, q_h_in)
        q = self.q_fc2(q_h)

        y = f.relu(self.i_fc1(obs))
        i_h_in = hidden_state_i.reshape(-1, self.args.rnn_hidden_dim)
        i_h = self.i_rnn(y, i_h_in)
        i = self.i_fc2(i_h)
        return q, q_h, i, i_h, f.log_softmax(i, dim=1)