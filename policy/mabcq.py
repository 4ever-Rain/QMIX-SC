import torch
import os
from network.base_net import BCRNN
from network.qmix_net import QMixNet
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time


class MABCQ:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度 3m地图，30个obs+7动作+3agent=40
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        # 神经网络
        self.eval_rnn = BCRNN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = BCRNN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMixNet(args)
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/qmix_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
        if args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.q_target_hidden = None

        self.i_eval_hidden = None
        self.i_target_hidden = None

        self.threshold = args.BCQ_threshold
        print('Init alg MABCQ')

        filename = (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        self.writer = SummaryWriter(self.args.result_dir + '/' + args.alg + '/' + args.map + '/logs_' + filename)
        self.writer.add_text("args", str(args))

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        # FIXME：在这里把函数展开写 拉开。
        q_evals, q_targets, i_loss= self.get_q_values(batch, max_episode_len)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
            avail_u_next = avail_u_next.cuda()

        # imt = imt_normal
        # 这一部分就是DQN的那个loss啥的
        # 得到target_q
        # TODO：把这里的target使用imt进行计算 需要对动作状态加mask，第一行是对不合法动作加了mask 需要再加一行对于没有达到阈值要求的动作加mask
        # with torch.no_grad():
        #     q_evals[avail_u_next == 0.0] = - 9999999
        #     imt = imt.exp()
        #     imt[avail_u_next == 0.0] = - 9999999
        #     imt = (imt/imt.max(3, keepdim=True)[0] > self.threshold).float()

        #     # Use large negative number to mask actions from argmax
        #     next_action = (imt * q_evals + (1 - imt) * -1e8).argmax(3, keepdim=True)
        #     q_targets = torch.gather(q_targets, dim=3, index=next_action).squeeze(3)
            
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_targets = q_targets.squeeze(3)
        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error



        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        q_loss = (masked_td_error ** 2).sum() / mask.sum()
        # print("i", i_loss)
        # print("q", q_loss)
        # print("reg", i.pow(2).mean())
        loss = q_loss + i_loss
        self.writer.add_scalar('i_loss', i_loss, global_step=train_step)
        self.writer.add_scalar('q_loss', q_loss, global_step=train_step)
        self.writer.add_scalar('Total_loss', loss, global_step=train_step)
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        # 96是因为一个batch的数据是32 然后有个三个agent 即 96
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        # imts, i_s = [], []
        i_loss = 0
        u, avail_u_next = batch['u'], batch['avail_u_next']
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.q_target_hidden = self.q_target_hidden.cuda()
                self.i_eval_hidden = self.i_eval_hidden.cuda()
                self.i_target_hidden = self.i_target_hidden.cuda()
                u = u.cuda()
                avail_u_next = avail_u_next.cuda()

            # 修改q_target的计算方式，并且 使用两个q的方法
            with torch.no_grad():
                q_eval, self.eval_hidden, _, self.i_eval_hidden, imt = self.eval_rnn(inputs, self.eval_hidden, self.i_eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
                imt = imt.exp()
                u_mask = (avail_u_next[:,transition_idx] == 0.0).view(episode_num * self.n_agents, -1)
                # imt[u_mask] = - 9999999
                q_eval[u_mask] = -9999999

                imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
                # Use large negative number to mask actions from argmax
                next_action = (imt * q_eval + (1 - imt) * -1e8).argmax(1, keepdim=True)
                q_target, self.q_target_hidden , _, self.i_target_hidden, _= self.target_rnn(inputs_next, self.q_target_hidden, self.i_target_hidden)
                
                q_target = q_target.gather(1, next_action)
            
             
                

            q_eval, self.eval_hidden, i, self.i_eval_hidden, imt = self.eval_rnn(inputs, self.eval_hidden, self.i_eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            i_loss += F.nll_loss(imt,u[:,transition_idx].reshape(-1)) + 1e-2 * i.pow(2).mean()
            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            imt = imt.view(episode_num, self.n_agents, -1)
            # i = i.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
            # imts.append(imt)
            # i_s.append(i)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        i_loss = i_loss / max_episode_len
        # imts= torch.stack(imts, dim=1)
        # i_s= torch.stack(i_s, dim=1)
        return q_evals, q_targets, i_loss

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.q_target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.i_eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.i_target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_mabcq_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
