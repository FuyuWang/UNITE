import math
import random

from torch.distributions import Categorical
from torch.distributions import Bernoulli
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Agent(object):
    def __init__(self, model_defs, num_pe=4096, max_pipe=64, feat_size=128, num_episodes=32, lr=1e-3):

        super().__init__()

        # constants
        self.num_pes = num_pe
        self.model_defs = model_defs
        self.num_episodes = num_episodes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # pe_space_min = math.log2(num_pe / num_pipe / num_pipe)
        # pe_space_max = math.log2(num_pe / num_pipe * num_pipe)
        # self.pe_space = np.power(2, np.arange(pe_space_min, pe_space_max+1))
        # print(self.pe_space)
        # pe_space_min = num_pe / 1024

        self.num_pipe = None
        # self.max_pipe = max_pipe
        self.feat_size = feat_size

        pipe_space = set()
        for i in range(1, model_defs.shape[0] + 1):
            if model_defs.shape[0] % i == 0:
                pipe_space.add(i)
            if pow(2, i) <= model_defs.shape[0]:
                pipe_space.add(pow(2, i))
        self.pipe_space = np.array(list(pipe_space))
        print(self.pipe_space)
        # self.pipe_space = np.array([1,2,4,8,16,32])
        self.max_pipe = self.pipe_space.max()
        # stride = 4
        # self.pipe_space = np.arange(1, max_pipe//stride+1)*stride
        # self.pe_space = np.arange(1, 2 * self.num_pipe) * self.num_pes / 2 / self.num_pipe
        # self.layer_pipe_space = np.arange(num_pipe)
        # pe_space = np.arange(1, self.max_pipe + 1, 2) * self.num_pes / 2 / self.max_pipe
        # pe_space = np.array([1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32]) * self.num_pes / 2 / self.max_pipe
        self.num_blocks = 64
        self.pe_space = np.array([1.5, 2, 3, 4]) * self.num_pes / 2 / self.num_blocks
        # self.pe_space = np.array([1.5, 2, 2.5, 3, 3.5, 4, 5, 6]) * self.num_pes / 2 / self.num_blocks
        # pe_space = np.arange(1, 2*64 + 1, 4) * self.num_pes / 2 / 64
        # pe_space = np.arange(1, 64+1) * self.num_pes / 64
        # self.pe_space = pe_space.astype(int)
        self.layer_pipe_space = np.arange(self.max_pipe)
        # in_size1 = 1 + num_pipe
        # in_size2 = num_pipe+model_defs.shape[1]
        in_size0 = self.model_defs.shape[0]*self.model_defs.shape[1]
        in_size1 = 1 + self.max_pipe
        in_size2 = self.max_pipe + model_defs.shape[1]

        self.a2c = A2CNet(self.pipe_space, self.pe_space, self.layer_pipe_space,
                          in_size0, in_size1, in_size2, self.feat_size, num_episodes).to(self.device)

        # init LSTM buffers with the number of the environments
        # self.a2c.set_recurrent_buffers(num_envs)

        # optimizer
        self.lr = lr
        self.optimizer = optim.Adam(self.a2c.parameters(), self.lr, betas=(0.9, 0.999))

        self.used_pes = np.zeros(self.num_episodes)
        # self.layer_pipe_mask = np.zeros((self.num_episodes, self.num_pipe))
        self.num_pe_mask = np.zeros((self.num_episodes, len(self.pe_space)))
        self.layer_pipe_mask = np.zeros((self.num_episodes, self.max_pipe))
        self.rewards = []
        self.log_probs = []
        self.log_prob_masks = []
        self.values = []
        self.entropies = []

    def reset(self):
        # self.num_pipe = num_pipe
        # self.pe_space = np.arange(1, 2 * self.num_pipe) * self.num_pe / 2 / self.num_pipe
        # self.layer_pipe_space = np.arange(num_pipe)
        # in_size1 = 1 + num_pipe
        # in_size2 = num_pipe + self.model_defs.shape[1]
        # self.a2c = A2CNet(self.pe_space, self.layer_pipe_space, in_size1, in_size2, self.feat_size).to(self.device)
        self.a2c.reset()
        self.used_pes = np.zeros(self.num_episodes)
        self.num_pe_mask = np.zeros((self.num_episodes, len(self.pe_space)))
        # self.layer_pipe_mask = np.zeros((self.num_episodes, self.num_pipe))
        self.layer_pipe_mask = np.zeros((self.num_episodes, self.max_pipe))
        self.rewards = []
        self.log_probs = []
        self.log_prob_masks = []
        self.values = []
        self.entropies = []

    def set_num_pipe(self, num_pipe):
        self.num_pipe = num_pipe
        # self.num_pe_mask = (np.expand_dims(self.pe_space, 0) < (self.num_pes / 2 / num_pipe)) * -1000000.
        pe_space = np.array([1.5, 2, 3, 4]) * self.num_pes / 2 / num_pipe
        # pe_space = np.array([1.5, 2, 2.5, 3, 3.5, 4, 5, 6]) * self.num_pes / 2 / self.num_pipe
        self.pe_space = pe_space.astype(int)
        # print(self.pe_space)
        self.layer_pipe_mask[:, self.num_pipe:self.max_pipe] = -1000000.

    def adjust_lr(self):
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.requires_grad:
                    param_group['lr'] = param_group['lr'] * 0.8

    def step(self, state, instruction):
        state = torch.from_numpy(state).type(torch.FloatTensor).to(self.device)
        # print(instruction)

        if instruction == 0:
            mask = torch.from_numpy(np.expand_dims(self.pipe_space, 0) > self.model_defs.shape[0]).to(
                self.device).float() * -1000000.
            action, log_prob, log_prob_mask, entropy, value = self.a2c(state, 'num_pipe', mask)
            action = action.cpu().numpy()
            action = self.pipe_space[action]
            self.set_num_pipe(action[0])
            log_prob = log_prob.repeat(self.num_episodes)
            log_prob_mask = log_prob_mask.repeat(self.num_episodes)
            entropy = entropy.repeat(self.num_episodes)
            value = value.repeat(self.num_episodes)
        elif 1 <= instruction <= self.num_pipe:
            # num_pe_min1 = (self.num_pes / 2 / self.num_pipe)
            # num_pe_min2 = ((self.pe_space <= num_pe_min1) * 1 * self.pe_space).max()
            # num_pe_min = min(num_pe_min1, num_pe_min2)
            # remained_pes = self.num_pes - self.used_pes - num_pe_min * (self.num_pipe - instruction)
            # print("instruction: ", instruction, num_pe_min, self.used_pes[0], self.num_pipe, remained_pes[0])
            remained_pes = self.num_pes - self.used_pes - self.pe_space[0] * (self.num_pipe - instruction)
            mask = torch.from_numpy(np.expand_dims(self.pe_space, 0) > np.expand_dims(remained_pes, 1)).to(self.device).float() * -1000000.
            # mask += torch.from_numpy(self.num_pe_mask).to(self.device)
            action, log_prob, log_prob_mask, entropy, value = self.a2c(state, 'resource', mask)
            action = action.cpu().numpy()
            # print(action[0])
            action = self.pe_space[action]
            # print(action[0])
            self.record_resource(action)
        else:
            mask = torch.from_numpy(self.layer_pipe_mask).to(self.device).float()
            action, log_prob, log_prob_mask, entropy, value = self.a2c(state, 'layer_pipe', mask)
            action = action.cpu().numpy()
            # action = self.layer_pipe_space[action]
            self.record_layer_pipe(action)
            if instruction % self.num_pipe == 0:
                self.layer_pipe_mask = np.zeros((self.num_episodes, self.max_pipe))
                self.layer_pipe_mask[:, self.num_pipe:self.max_pipe] = -1000000.
        return action, log_prob, log_prob_mask, entropy, value

    def record_resource(self, action):
        self.used_pes += action

    def record_layer_pipe(self, action):
        self.layer_pipe_mask[np.arange(self.num_episodes), action] = -1000000.

    def record_reward(self, reward, log_prob, log_prob_mask, entropy, value):
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.log_prob_masks.append(log_prob_mask)
        self.entropies.append(entropy)
        self.values.append(value)

    def backward(self):
        CLIPPING_MODEL = 5
        loss = self.compute_policy_loss()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.a2c.parameters(), CLIPPING_MODEL)
        self.optimizer.step()

    def compute_policy_loss(self, filter=False):
        self.rewards = np.stack(self.rewards, axis=0)
        self.log_probs = torch.stack(self.log_probs, dim=0)
        self.log_prob_masks = torch.stack(self.log_prob_masks, dim=0)
        self.entropies = torch.stack(self.entropies, dim=0)
        self.values = torch.stack(self.values, dim=0)

        GAMMA = 0.99
        dis_rewards = []
        batch_size = self.log_probs.size(1)

        # batch_masks = torch.from_numpy(self.info).to(self.log_probs.device)
        # fail_idx = []
        # for i in range(batch_size):
        #     if info[i] < 0:
        #         fail_idx.append(i)
        # if len(fail_idx) > 4:
        #     fail_idx = random.sample(fail_idx, 4)
        # batch_masks[fail_idx] = 1.

        R = np.zeros(batch_size)
        for r in self.rewards[::-1]:
            R = r + GAMMA * R
            dis_rewards.insert(0, R)
        dis_rewards = torch.from_numpy(np.array(dis_rewards)).to(self.log_probs.device)

        # advantage = dis_rewards - self.values
        # policy_loss = (-1 * self.log_probs * self.log_prob_masks * advantage.detach()).mean(dim=0)
        # value_loss = advantage.pow(2).mean(dim=0)
        #
        # value_coeff = 0.5
        # entropy_coeff = 0.02
        # # loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropies.sum(dim=0)
        # loss = policy_loss + value_coeff * value_loss
        # print(policy_loss, value_loss)
        # return loss.mean()

        policy_loss = dis_rewards * (-1 * self.log_probs * self.log_prob_masks)
        return policy_loss.mean()


def init_weights(m):
    if type(m) == nn.LSTMCell:
        nn.init.orthogonal_(m.weight_hh)
        nn.init.orthogonal_(m.weight_ih)
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


class A2CNet(nn.Module):
    def __init__(self, pipe_space, pe_space, layer_pipe_space, in_size0, in_size1, in_size2, feat_size=128, num_episodes=32):
        super(A2CNet, self).__init__()

        # dim_length = 7
        # self.dim_encoder = nn.Sequential(
        #     nn.Linear(dim_length, dim_length*in_size),
        #     nn.ReLU(),
        #     nn.Linear(dim_length*in_size, feat_size),
        #     nn.ReLU(),
        #     nn.Linear(feat_size, feat_size),
        #     nn.ReLU(),
        # )
        self.pipe_space = pipe_space
        self.pe_space = pe_space
        self.layer_pipe_space = layer_pipe_space

        self.in_size0 = in_size0
        self.in_size1 = in_size1
        self.in_size2 = in_size2
        self.feat_size = feat_size
        self.num_episodes = num_episodes

        self.num_pipe_encoder = nn.Sequential(
            nn.Linear(in_size0, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
        )

        self.num_pipe_decoder = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, len(pipe_space)),
        )

        self.num_pe_encoder = nn.Sequential(
            nn.Linear(in_size1, in_size1 * 10),
            nn.ReLU(),
            nn.Linear(in_size1 * 10, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
        )

        self.num_pe_decoder = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, len(pe_space)),
        )

        self.layer_pipe_encoder = nn.Sequential(
            nn.Linear(in_size2, in_size2 * 10),
            nn.ReLU(),
            nn.Linear(in_size2 * 10, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
        )

        self.layer_pipe_decoder = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, len(layer_pipe_space)),
        )

        self.critic = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, 1),
        )

        self.lstm = torch.nn.LSTMCell(feat_size, feat_size)

        self.lstm_value = None

        self.init_weight()

    def reset(self):
        self.lstm_value = self.init_hidden()

    def init_weight(self):
        self.apply(init_weights)

    def init_hidden(self):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_episodes, self.feat_size),
                weight.new_zeros(self.num_episodes, self.feat_size))

    # def set_tile_temperature(self, temp):
    #     self.tile_temperature = temp
    #
    # def set_order_temperature(self, temp):
    #     self.order_temperature = temp
    #
    # def set_parallel_temperature(self, temp):
    #     self.parallel_temperature = temp

    def forward(self, state, instruction, mask=None):
        if instruction == 'num_pipe':
            feat = self.num_pipe_encoder(state)
            # h, x = self.lstm(feat, self.lstm_value)
            # self.lstm_value = (h, x)
            h = feat
            num_pe_score = self.num_pipe_decoder(h) + mask
            num_pe_prob = F.softmax(num_pe_score, dim=-1)
            num_pe_density = Categorical(num_pe_prob)
            num_pe_action = num_pe_density.sample()
            num_pe_log_prob = num_pe_density.log_prob(num_pe_action)
            action = num_pe_action
            log_prob = num_pe_log_prob
            entropy = num_pe_density.entropy()
        elif instruction == 'resource':
            pad = state.new_zeros(state.size(0), self.in_size1 - state.size(1))
            state = torch.cat([state, pad], dim=-1)
            feat = self.num_pe_encoder(state)
            # h, x = self.lstm(feat, self.lstm_value)
            # self.lstm_value = (h, x)
            h = feat
            num_pe_score = self.num_pe_decoder(h) + mask
            num_pe_prob = F.softmax(num_pe_score, dim=-1)
            num_pe_density = Categorical(num_pe_prob)
            num_pe_action = num_pe_density.sample()
            num_pe_log_prob = num_pe_density.log_prob(num_pe_action)
            action = num_pe_action
            log_prob = num_pe_log_prob
            entropy = num_pe_density.entropy()
        else:
            pad = state.new_zeros(state.size(0), self.in_size2 - state.size(1))
            state = torch.cat([state, pad], dim=-1)
            feat = self.layer_pipe_encoder(state)
            # h, x = self.lstm(feat, self.lstm_value)
            # self.lstm_value = (h, x)
            h = feat
            layer_pipe_score = self.layer_pipe_decoder(h) + mask
            layer_pipe_prob = F.softmax(layer_pipe_score, dim=-1)
            layer_pipe_density = Categorical(layer_pipe_prob)
            layer_pipe_action = layer_pipe_density.sample()
            layer_pipe_log_prob = layer_pipe_density.log_prob(layer_pipe_action)
            action = layer_pipe_action
            log_prob = layer_pipe_log_prob
            entropy = layer_pipe_density.entropy()

        value = self.critic(h)

        log_prob_mask = ((mask == 0).sum(dim=-1) > 1).float()

        return action, log_prob, log_prob_mask, entropy, value.squeeze()
