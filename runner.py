import numpy as np
import os
from common.rollout import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, env, args):
        self.env = env

        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if not args.evaluate :
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        self.buffer_path = self.save_path + '/buffer'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.buffer_path):
            os.makedirs(self.buffer_path)
        if self.args.buffer_with_run:
            if not os.path.exists(self.save_path + '/buffer_run'):
                os.makedirs(self.save_path + '/buffer_run')

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        while train_steps < self.args.train_epoch:
            print('Run {}, train_steps {}'.format(num, train_steps))
            if train_steps // self.args.evaluate_frq > evaluate_steps:
                win_rate, episode_reward = self.evaluate()
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.plt(num)
                evaluate_steps += 1
            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            # 比如3m地图中 o.shape(1,60,3,30)有60的timestep 具体是30的维度； 具体的和buufer里面的的定义一模一样
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            # if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
            #     self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
            #     train_steps += 1
            # else:
            self.buffer.store_episode(episode_batch)
            
            # 训练循环一次 
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                self.agents.train(mini_batch, train_steps)
                train_steps += 1
        if self.args.buffer_with_run:
            self.buffer.save(self.save_path + '/buffer_run')
        win_rate, episode_reward = self.evaluate()
        print('win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt(num)

        return win_rate, episode_reward

    def run_offline(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        # Load offline buffer
        self.buffer.load(self.buffer_path)
        while train_steps < self.args.train_epoch:
            print('Run {}, train_steps {}'.format(num, train_steps))
            if train_steps // self.args.evaluate_frq > evaluate_steps:
                # 每5个epoch评估一次表现
                win_rate, episode_reward = self.evaluate()
                print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.plt(num)
                evaluate_steps += 1
           
            # 训练循环一次 
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                self.agents.train(mini_batch, train_steps)
                train_steps += 1
        win_rate, episode_reward = self.evaluate()
        print('win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt(num)

        return win_rate, episode_reward

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def generate_buffer(self):
        time_steps = 0
        for _ in range(self.args.buffer_size):
            episode, _, _, steps = self.rolloutWorker.generate_episode(evaluate=True)
            time_steps += steps
            self.buffer.store_episode(episode)
            print('Run, time_steps {}'.format(time_steps))
            print('Run, buffer_size {}'.format(self.buffer.current_size))
            
        self.buffer.save(self.buffer_path)
        print("===============================")
        print("Total buffer:", self.buffer.current_size)
        print("Total time_steps:", time_steps)
        print("Genrate buffer successfully!!")
        print("===============================")
        win_rate, episode_reward = self.evaluate()
        return win_rate, episode_reward

    def plt(self, num):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('Epoch*{}'.format(self.args.evaluate_frq))
        plt.ylabel('win_rates')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('Epoch*{}'.format(self.args.evaluate_frq))
        plt.ylabel('episode_rewards')

        if self.args.offline:
            plt.savefig(self.save_path + '/off_plt_{}.png'.format(num), format='png')
            np.save(self.save_path + '/off_win_rates_{}'.format(num), self.win_rates)
            np.save(self.save_path + '/off_episode_rewards_{}'.format(num), self.episode_rewards)
        else:
            plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
            np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
            np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        plt.close()









