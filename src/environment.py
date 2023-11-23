
from subprocess import Popen, PIPE
import pandas as pd
import os
import random
import pickle
import copy
import os, sys
import math
import numpy as np
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)

df_dict = {1:"dla", 2:"shi", 3:"eye"}
m_type_dicts = {1:"CONV", 2:"DSCONV", 3:"CONV", 4:"TRCONV"}


class MaestroEnvironment(object):
    def __init__(self, model_defs, input_size=32, dataflow="dla",
                 num_pe=4096, l1_size=8000, l2_size=8000, bandwidth=256,
                 max_pipe=64, fitness='latency', num_episodes=None):
        super(MaestroEnvironment,self).__init__()
        dst_path = "../cost_model/maestro"

        maestro = dst_path
        self._executable = "{}".format(maestro)

        self.max_pipe = max_pipe
        self.num_pipe = None
        self.total_steps = 1+max_pipe+model_defs.shape[0]
        self.fitness = fitness
        self.num_episodes = num_episodes
        self.mode = 0
        # self.solutions = np.zeros((num_episodes, self.total_steps), dtype=np.int32)
        # self.solutions[:, 0] = num_pipe
        self.solutions = None
        self.sol_record = []
        self.best_sol = None
        self.min_reward = None
        self.best_reward = float('-inf')
        self.input_size = input_size
        self.epoch = 0

        self.dataflow = dataflow
        self.constraint_value = np.array([num_pe, l1_size, l2_size, bandwidth])
        self.total_used_constraints = np.array([0., 0., 0.])

        self.model_defs = model_defs
        self.model_defs_saved = copy.deepcopy(model_defs)
        model_bound = np.max(model_defs, axis=0, keepdims=True)
        self.model_defs_norm = model_defs/model_bound
        self.model_defs_norm_saved = copy.deepcopy(self.model_defs_norm)

        # self.model2draw = {}
        # self.model2baseline = {}
        # self.draw = self.get_draw(self.model_name, self.model_defs)
        # self.model_defs = self.model_defs_saved[self.draw]
        # self.model_defs_norm = self.model_defs_norm_saved[self.draw]

    # def shuffle_model(self):
    #     draw = np.random.permutation(self.total_step)
    #     self.model_defs = self.model_defs_saved[draw]
    #     self.model_defs_norm = self.model_defs_norm_saved[draw]
    #     self.draw = draw

    def set_fitness(self, fitness="latency"):
        self.fitness = fitness

    def set_constraint(self, constraint="buffer_size"):
        self.constraint = constraint

    def reset(self):
        """

        """
        self.mode = 0
        self.total_used_constraints = np.array([0., 0., 0.])

        # self.num_pipe = num_pipe
        # self.total_steps = 1 + self.num_pipe + self.model_defs.shape[0]
        # self.solutions = np.zeros((self.num_episodes, self.total_steps), dtype=np.int32)
        # self.solutions[:, 0] = self.num_pipe
        self.total_steps = 1 + self.max_pipe + self.model_defs.shape[0]
        self.solutions = None

        # state = self.solutions[:, 0:self.num_pipe+1]
        # state = state / self.constraint_value[0]
        state = np.reshape(self.model_defs_norm,(1, self.model_defs_norm.shape[0]*self.model_defs_norm.shape[1]))
        return state

    def norm_state(self, T):
        T[:-1] = (T[:-1] - 0.5) * 2
        return T

    def get_reward(self, sol):
        # print(sol, len(sol))
        # sol = sol.astype(int)
        num_pipe = sol[0]
        num_pes = sol[1:num_pipe+1]
        layer_pipe = sol[num_pipe+1:]
        # print(sol, num_pes, layer_pipe, num_pes.sum())
        total_mac = []
        total_runtime = []
        total_energy = []
        total_constraint = []

        num_layers = self.model_defs.shape[0]
        groups = num_layers // num_pipe + 1
        for g in range(groups):
            group_mac = np.zeros(num_pipe)
            group_runtime = np.zeros(num_pipe)
            group_energy = np.zeros(num_pipe)
            group_constraint = np.array([0., 0., 0.])
            for i in range(num_pipe):
                if g*num_pipe + i >= num_layers:
                    break
                num_pe = num_pes[layer_pipe[g*num_pipe + i]]
                bandwidth = int(num_pe / self.constraint_value[0] * self.constraint_value[-1])
                dimension = self.model_defs[g*num_pipe + i]
                runtime, energy, mac, l1_size, l2_size = self.observe_maestro(num_pe, bandwidth, dimension)
                group_mac[i] = mac
                group_runtime[i] = runtime
                group_energy[i] = energy
                group_constraint = group_constraint + np.array([num_pe, l1_size, l2_size])

            #     print(num_pe, dimension)
            #
            # print(group_runtime)
            group_runtime = np.array(group_runtime)
            group_runtime_sum = (group_runtime.sum() + (self.input_size - 1) * group_runtime.max())

            total_mac.append(group_mac)
            total_runtime.append(group_runtime_sum)
            total_energy.append(group_energy)
            total_constraint.append(group_constraint)
        return -1. * np.array(total_runtime).sum()

    def step(self, action):
        done = 0
        self.mode += 1

        if self.mode == 1:
            self.num_pipe = action[0]
            self.total_steps = 1 + self.num_pipe + self.model_defs.shape[0]
            self.solutions = np.zeros((self.num_episodes, self.total_steps), dtype=np.int32)
            self.solutions[:, 0] = self.num_pipe
            state = self.solutions[:, 0:self.num_pipe + 1]
            state = state.astype(float)
            state[:, 0] /= self.max_pipe
            state[:, 1:] /= self.constraint_value[0]
        elif 1 < self.mode <= self.num_pipe:
            self.solutions[:, self.mode - 1] = action
            state = self.solutions[:, 0:self.num_pipe + 1]
            state = state.astype(float)
            state[:, 0] /= self.max_pipe
            state[:, 1:] /= self.constraint_value[0]
        elif self.num_pipe < self.mode < self.total_steps:
            self.solutions[:, self.mode-1] = action
            state = np.zeros((self.num_episodes, self.num_pipe+self.model_defs.shape[1]))
            state[:, 0:self.num_pipe] = self.solutions[:, 1:self.num_pipe + 1] / self.constraint_value[0]

            # if self.model_defs_saved[self.mode - self.num_pipe - 1][3] == 224:
            if self.model_defs_saved[self.mode - self.num_pipe - 1][3] > self.model_defs_saved[self.mode - self.num_pipe - 1][0] and \
                    self.model_defs_saved[self.mode - self.num_pipe - 1][-1] != 2:
                model_defs_norm = copy.deepcopy(self.model_defs_norm[self.mode - self.num_pipe - 1])
                model_defs_norm[0] = self.model_defs_norm[self.mode - self.num_pipe - 1][2]
                model_defs_norm[1] = self.model_defs_norm[self.mode - self.num_pipe - 1][3]
                model_defs_norm[2] = self.model_defs_norm[self.mode - self.num_pipe - 1][0]
                model_defs_norm[3] = self.model_defs_norm[self.mode - self.num_pipe - 1][1]
            else:
                state[:, self.num_pipe:] = self.model_defs_norm[self.mode - self.num_pipe - 1]
        else:
            self.solutions[:, self.mode-1] = action
            done = 1
            state = None

        if self.mode == self.total_steps:
            pool = Pool(min(action.shape[0], cpu_count()))
            return_list = pool.map(self.get_reward, self.solutions)
            reward_saved = copy.deepcopy(np.array(return_list))
            if self.min_reward is None:
                self.min_reward = reward_saved.min()
            self.min_reward = min(self.min_reward, reward_saved.min())
            reward = reward_saved - self.min_reward
        else:
            reward = np.zeros(self.num_episodes)
            reward_saved = np.zeros(self.num_episodes).fill(self.min_reward)

        # print(state.shape)
        return self.solutions, state, reward, reward_saved, done

    def load_chkpt(self, chkpt):
        self.reward_rec = chkpt["reward_rec"]
        self.best_reward = chkpt["best_reward"]
        self.best_rewards= chkpt["best_rewards"]
        self.best_sol= chkpt["best_sol"]
        self.worst_reward = chkpt["worst_reward"]

    def get_chkpt(self):
        return {
            "best_sol": self.best_sol,
            "best_reward": self.best_reward
        }

    def save_chkpt(self, chkpt_file=None):
        if chkpt_file is None:
            chkpt_file = self.chkpt_file
        chkpt = self.get_chkpt()
        with open(chkpt_file, "wb") as fd:
            pickle.dump(chkpt, fd)

    def observe_maestro(self, num_pe, bandwidth, dimension):
        m_file = "{}".format(random.randint(0, 2 ** 31))
        m_type = m_type_dicts[int(dimension[-1])]
        # if dimension[3] == 224:
        if dimension[3] > dimension[0]:
            if m_type == "DSCONV":
                dataflow = 'dsconv_shi'
            else:
                dataflow = 'shi'
            KTileSz = 1
            CTileSz = 1
            ClusterSz = min(num_pe, dimension[3] // 2)
        else:
            if m_type == "DSCONV":
                dataflow = 'dpt'
            else:
                dataflow = 'dla'
            KTileSz = 1
            CTileSz = 1
            ClusterSz = min(num_pe, dimension[1])

        with open("../data/dataflow/{}.m".format(dataflow), "r") as fd:
            # with open("../data/dataflow/dpt.m", "r") as fdpt:
                with open("{}.m".format(m_file), "w") as fo:
                    fo.write("Constant KTileSz {};\n".format(KTileSz))
                    fo.write("Constant CTileSz {};\n".format(CTileSz))
                    fo.write("Constant ClusterSz {};\n".format(ClusterSz))
                    fo.write("Network {} {{\n".format(0))
                    fo.write("Layer {} {{\n".format(m_type))
                    fo.write("Type: {}\n".format(m_type))
                    fo.write(
                        "Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(
                            *dimension[:6]))
                    # if m_type == "CONV" or m_type == "TRCONV":
                    #     fd.seek(0)
                    #     fo.write(fd.read())
                    # else:
                    #     fdpt.seek(0)
                    #     fo.write(fdpt.read())
                    fd.seek(0)
                    fo.write(fd.read())
                    fo.write("}\n")
                    fo.write("}")

        # with open("{}.m".format(m_file), "r") as fr:
        #     lines = fr.readlines()
        #     print(lines)

        os.remove("./{}.csv".format(m_file)) if os.path.exists("./{}.csv".format(m_file)) else None
        command = [self._executable,
                   "--Mapping_file={}.m".format(m_file),
                   "--full_buffer=false", "--noc_bw_cstr={}".format(bandwidth),
                   "--noc_hops=1", "--noc_hop_latency=1",
                   "--noc_mc_support=true", "--num_pes={}".format(num_pe),
                   "--num_simd_lanes={}".format(1), "--l1_size_cstr=81920000",
                   "--l2_size_cstr=819200000", "--print_res=false", "--print_res_csv_file=true", "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]

        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        process.wait()
        try:
            df = pd.read_csv("./{}.csv".format(m_file))
            # print(num_pe, dimension, df[' NoC BW (Elements/cycle)'])
            layer_name = df[" Layer Number"]
            runtime = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            throughput = np.array(df[" Throughput (MACs/Cycle)"]).reshape(-1, 1)
            energy = np.array(df[" Activity count-based Energy (nJ)"]).reshape(-1, 1)
            area = np.array(df[" Area"]).reshape(-1, 1)
            power = np.array(df[" Power"]).reshape(-1, 1)
            l1_size = np.array(df[" L1 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            l2_size = np.array(df["  L2 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            mac = np.array(df[" Num MACs"]).reshape(-1, 1)
            os.remove("./{}.csv".format(m_file))  if os.path.exists("./{}.csv".format(m_file)) else None
            os.remove("./log.txt") if os.path.exists("./log.txt") else None
            # print(area)
            # self.observation = [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power]]
            # return runtime, energy, mac, l1_size, l2_size
            return np.sum(runtime), np.sum(energy), np.sum(mac), np.max(l1_size), np.max(l2_size)
        except Exception as e:
            print(e)
            print("+"*20)
            print(num_pe, dimension, dataflow)
            return float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')
