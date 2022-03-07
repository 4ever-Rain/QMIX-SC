from unittest import result
from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
import time
import os
import json

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


    train_nums = 1

    starttime = time.time()
    filename = (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    
    for _ in range(train_nums):
        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        runner = Runner(env, args)

        if args.generate_buffer:
            # Generate buffer
            if not args.load_model:
                raise Exception("No model load!")
            print("================Genarate buffer==============")
            win_rate, episode_reward = runner.generate_buffer()
            print('The buffer win rate of {} is {}'.format(args.alg, win_rate))
            print('The buffer episode reward of {} is  {}'.format(args.alg, episode_reward))
            break
        elif args.offline and not args.evaluate:
            # Offline alg
            print("============ Using offline alg ===========")
            wr, er = runner.run_offline(args.train_no)
            win_result_list.append(wr)
            episode_reward_list.append(er)
        elif not args.evaluate:
            # Online alg (normal)
            print("============ Using On-line alg ===========")
            wr, er = runner.run(args.train_no)
            win_result_list.append(wr)
            episode_reward_list.append(er)
        elif args.evaluate:
            # Evaluate
            print("============ Evaluate ===========")
            win_rate, episode_reward = runner.evaluate()
            print('The win rate of {} is {}'.format(args.alg, win_rate))
            print('The episode reward of {} is {}'.format(args.alg, episode_reward))
            win_result_list.append(win_rate)
            episode_reward_list.append(episode_reward)
            break
        else:
            raise Exception("Error args! Not evl, not train online, not train offline!")
        env.close()

    # 记录所有的实验数据
    endtime = time.time()
    dtime = endtime - starttime

    if not os.path.exists("/home/ubuntu/SC2_result/config"):
        os.makedirs("/home/ubuntu/SC2_result/config")
    filepath = "/home/ubuntu/SC2_result/config/" + filename + ".txt"
    configlist = {
        "start_time" : filename,
        "end_time" : time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
        "running_time" : dtime,
        "train_nums" : train_nums,
        "alg" : args.alg,
        "map" : args.map,
        "n_step" : args.n_steps,
        "win_rate" : win_result_list,
        "episode_reward" : episode_reward_list,
    }
    import json
    with open(filepath, 'w') as fp:
        fp.write(json.dumps(configlist, indent=4))
        fp.write(json.dumps(vars(args), indent=4))

    fp.close()

