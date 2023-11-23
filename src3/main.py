import copy
import argparse
import random
import pandas as pd
import glob
import os, sys
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

from environment import MaestroEnvironment
from agent import Agent

script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #  torch.backends.cudnn.deterministic = True


def rl_search(model_defs):
    # num_pipe_space = [4, 8, 12, 16, 20, 24, 28, 32]
    num_pipe = 6
    # pe_alloc = np.array([6301, 5041, 6301, 5041, 7561, 3780, 6301, 6301, 3780, 3780, 3780, 3780, 3780])
    # pe_alloc = np.array([13653,  8192, 10922, 16384,  8192,  8192])
    pe_alloc = np.array([512,  512,  768,  640,  384, 1024,  384,  896,  384,  384,  384,  384,  384, 384, 384])
    pipe_epochs = 20
    num_episodes = 32
    best_reward = float('-inf')

    # for num_pipe in num_pipe_space:
    # for num_pipe in [52]:
    #     print(num_pipe)
    agent = Agent(model_defs=model_defs, num_pe=opt.num_pe, num_pipe=num_pipe,
                  feat_size=256, num_episodes=num_episodes, lr=5e-4)
    env = MaestroEnvironment(model_defs=model_defs, input_size=opt.input_size, dataflow="dla",
                             num_pe=opt.num_pe, l1_size=opt.l1_size, l2_size=opt.l2_size, bandwidth=opt.bandwidth,
                             num_pipe=num_pipe, pe_alloc=pe_alloc, fitness='latency', num_episodes=num_episodes)
    for epoch in range(pipe_epochs):
        state = env.reset()
        agent.reset()
        # print(datetime.now().time())
        for t in range(1 + num_pipe, env.total_steps):
            action, log_prob, log_prob_mask, entropy, value = agent.step(state, t)
            solutions, state, reward, reward_saved = env.step(action)
            agent.record_reward(reward, log_prob, log_prob_mask, entropy, value)
            # print(reward)

        best_idx = np.argmax(reward_saved)
        if reward_saved[best_idx] > best_reward:
            best_reward = reward_saved[best_idx]
            # agent_chkpt['best_actor'] = actor.state_dict()
            # agent_chkpt['best_reward'] = best_reward
            # agent_chkpt['best_latency'] = latency[best_idx]
            # agent_chkpt['best_power'] = power[best_idx]
            # agent_chkpt['best_energy'] = energy[best_idx]
            # agent_chkpt['best_area'] = area[best_idx]
            # agent_chkpt['best_resource'] = resource[best_idx]
            # agent_chkpt['best_sol'] = sol[best_idx]
            # agent_chkpt['best_state'] = state[best_idx]
            # print("Epoch {}, Best Reward: {}, Best Sol: {}".format(ep, best_reward, latency[best_idx], power[best_idx], sol[best_idx]))
            # print(f"Epoch {epoch}, Best Reward: { best_reward, latency[best_idx], power[best_idx], area[best_idx]}, "
            #       f"Best Sol: {resource[best_idx], sol[best_idx]}")
            print(f"Epoch {epoch}, Best Reward: {best_reward}",
                  f"Best Sol: {solutions[best_idx]}"
                  )

        # agent_chkpt['best_reward_record'].append(agent_chkpt['best_reward'])
        # agent_chkpt['best_latency_record'].append(agent_chkpt['best_latency'])
        # agent_chkpt['best_power_record'].append(agent_chkpt['best_power'])
        # agent_chkpt['best_energy_record'].append(agent_chkpt['best_energy'])
        # agent_chkpt['best_area_record'].append(agent_chkpt['best_area'])
        # agent_chkpt['best_sols'].append(agent_chkpt['best_sol'])
        # log_str = f"Epoch {ep}, Best Reward: {best_reward, agent_chkpt['best_latency'], agent_chkpt['best_power'], agent_chkpt['best_area']}, " \
        #           f"Best Sol: {agent_chkpt['best_resource'], agent_chkpt['best_sol']}\n"
        log_str = f"Best Reward: {best_reward}\n"
        print(log_str)
        epf.write(log_str)
        epf.flush()
        # policy_loss /= num_episodes
        agent.backward()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness', type=str, default="latency", help='objective fitness')
    parser.add_argument('--input_size', type=int, default=1, help='number of inputs')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--outdir', type=str, default="outdir", help='output directiory')
    parser.add_argument('--model', type=str, default="vgg16", help='Model to run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_pe', type=int, default=4096)
    parser.add_argument('--l1_size', type=int, default=8000)
    parser.add_argument('--l2_size', type=int, default=8000)
    parser.add_argument('--bandwidth', type=int, default=256)

    opt = parser.parse_args()
    m_file_path = "../data/model/"
    # for input_size in [1, 4, 16, 64, 128, 256, 512]:
    # for input_size in [64, 256, 512]:
    #     # for model in ['resnet18', 'UNet', 'mobilenet_v2']:
    for model in ['resnet18']:
        m_file = os.path.join(m_file_path, model + ".csv")
        df = pd.read_csv(m_file)
        model_defs = df.to_numpy()

        outdir = opt.outdir
        outdir = os.path.join("../", outdir)

        exp_name = "{}_{}_inputs-{}_PE-{}_L1-{}_l2-{}_EPOCH-{}".format(model, opt.fitness, opt.input_size,
                                                                       opt.num_pe, opt.l1_size, opt.l2_size, opt.bandwidth,
                                                                       opt.epochs)

        outdir_exp = os.path.join(outdir, exp_name)
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(outdir_exp, exist_ok=True)

        try:
            log_file = os.path.join(outdir_exp, "result.log")
            epf = open(log_file, 'a')
            set_seed(opt.seed)
            rl_search(model_defs)
            # dimension_to_key = ','.join(str(j) for j in dimension)
            # if dimension_to_key in dim2chkpt:
            #     agent_chkpt = dim2chkpt[dimension_to_key]
            #     pickle.dump(agent_chkpt, open(os.path.join(outdir_exp, 'agent_chkpt.plt'), 'wb'))
            #     print("repeated")
            # else:
            #     chkpt_file_t = "{}".format("result")
            #     log_file = os.path.join(outdir_exp, chkpt_file_t + ".log")
            #     epf = open(log_file, 'a')
            #     # dimension[2] /= dimension[6]
            #     # dimension[3] /= dimension[6]
            #     print(dimension, dimension_to_key)
            #     try:
            #         set_seed(opt.seed)
            #         if opt.learning:
            #             print(opt.learning)
            #             new_state_dict, agent_chkpt = train(dimension, new_state_dict)
            #         else:
            #             new_state_dict, agent_chkpt = train(dimension)
            #         dim2chkpt[dimension_to_key] = agent_chkpt
            #         torch.save(new_state_dict, os.path.join(outdir_exp, 'state_dict.plt'))
            #         pickle.dump(agent_chkpt, open(os.path.join(outdir_exp, 'agent_chkpt.plt'), 'wb'))
        finally:
            for f in glob.glob("*.m"):
                os.remove(f)
            for f in glob.glob("*.csv"):
                os.remove(f)

            # print('epoch end : ', datetime.now())

