import numpy as np
import os
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
# save_path = "/home/ubuntu/QMIX-SC/result/mabcq/3m/buffer_15k_mix/3m_r.npy"
# reward = np.load(save_path)
# print(reward.sum(axis=1).mean())
# print(reward.shape)


if __name__ == '__main__':
    win_result_list = []
    episode_reward_list = []

    # get args
    args = get_common_args()
    if args.alg.find('coma') > -1:
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)


    folder1 = "/home/ubuntu/QMIX-SC/result/qmix/3m/buffer_5k_3kst_15.07"
    folder2 = "/home/ubuntu/QMIX-SC/result/qmix/3m/buffer_5k_5kst_9.1828"
    folder3 = "/home/ubuntu/QMIX-SC/result/mabcq/3m/buffer_10k_19.347"
    folder4 = "/home/ubuntu/QMIX-SC/result/mabcq/3m/buffer_15k_mix"

    if not os.path.exists(folder4):
        os.makedirs(folder4)


    map="3m"
    size = 15000

    env = StarCraft2Env(map_name=map,
                        step_mul=args.step_mul,
                        difficulty=args.difficulty,
                        game_version=args.game_version,
                        replay_dir=args.replay_dir)
    env_info = env.get_env_info()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    state_shape = env_info["state_shape"]
    obs_shape = env_info["obs_shape"]
    episode_limit = env_info["episode_limit"]

    key_list = ['o','u','s','r','o_next','s_next','avail_u','avail_u_next','u_onehot','padded','terminated']
    buffers = {'o': np.empty([size, episode_limit, n_agents, obs_shape]),
            'u': np.empty([size, episode_limit, n_agents, 1]),
            's': np.empty([size, episode_limit, state_shape]),
            'r': np.empty([size, episode_limit, 1]),
            'o_next': np.empty([size, episode_limit, n_agents, obs_shape]),
            's_next': np.empty([size, episode_limit, state_shape]),
            'avail_u': np.empty([size, episode_limit, n_agents, n_actions]),
            'avail_u_next': np.empty([size, episode_limit, n_agents, n_actions]),
            'u_onehot': np.empty([size, episode_limit, n_agents, n_actions]),
            'padded': np.empty([size, episode_limit, 1]),
            'terminated': np.empty([size, episode_limit, 1])
            }

    for key in key_list:
        t1 = np.load(folder1 + f"/{map}_{key}.npy")
        t2 = np.load(folder2 + f"/{map}_{key}.npy")
        t3 = np.load(folder3 + f"/{map}_{key}.npy")[:5000]
        t = buffers[key]
        t[:5000] = t1
        t[5000:10000] = t2
        t[10000:] = t3

        if key == 'r':
            print("reward mean:", t.sum(axis=1).mean())

        np.save(folder4 + f"/{map}_{key}.npy", t)
        print("save ", f"/{map}_{key}.npy")
    

    print("==================Finish!===================")









