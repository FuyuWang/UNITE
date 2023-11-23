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
    def __init__(self, model_defs, num_pe=4096, bandwidth=4096, num_pipe=8, feat_size=128, num_episodes=32, lr=1e-3):

        super().__init__()

        # constants
        self.num_pes = num_pe
        self.bandwidth = bandwidth
        self.model_defs = model_defs
        self.num_episodes = num_episodes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # pe_space_min = math.log2(num_pe / num_pipe / num_pipe)
        # pe_space_max = math.log2(num_pe / num_pipe * num_pipe)
        # self.pe_space = np.power(2, np.arange(pe_space_min, pe_space_max+1))
        # print(self.pe_space)
        # pe_space_min = num_pe / 1024

        self.num_pipe = num_pipe
        self.feat_size = feat_size

        # self.pe_space = (np.array([1, 1.5, 2, 3, 4, 6, 8]) * self.num_pes / 2 / self.num_pipe).astype(int)
        self.pe_space = (np.array([1.5, 2, 2.5, 3, 3.5, 4]) * self.num_pes / 2 / self.num_pipe).astype(int)
        self.bw_space = (np.array([1.5, 2, 2.5, 3, 3.5, 4]) * self.bandwidth / 2 / self.num_pipe).astype(int)
        # self.pe_space = (np.arange(1, 16 + 1) * self.num_pes / 16).astype(int)
        self.layer_pipe_space = np.arange(num_pipe)

        in_size1 = 1 + num_pipe
        in_size2 = num_pipe+model_defs.shape[1]

        self.a2c = A2CNet(self.pe_space, self.bw_space, self.layer_pipe_space, in_size1, in_size2, self.feat_size, num_episodes).to(self.device)

        # init LSTM buffers with the number of the environments
        # self.a2c.set_recurrent_buffers(num_envs)

        # optimizer
        self.lr = lr
        self.optimizer = optim.Adam(self.a2c.parameters(), self.lr, betas=(0.9, 0.999))

        self.used_pes = np.zeros(self.num_episodes)
        self.layer_pipe_mask = np.zeros((self.num_episodes, self.num_pipe))
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
        self.layer_pipe_mask = np.zeros((self.num_episodes, self.num_pipe))
        self.rewards = []
        self.log_probs = []
        self.log_prob_masks = []
        self.values = []
        self.entropies = []

    def step(self, state, instruction):
        state = torch.from_numpy(state).type(torch.FloatTensor).to(self.device)
        if 1 <= instruction <= self.num_pipe:
            remained_pes = self.num_pes - self.used_pes - self.pe_space[0] * (self.num_pipe - instruction)
            mask = torch.from_numpy(np.expand_dims(self.pe_space, 0) > np.expand_dims(remained_pes, 1)).to(self.device).float() * -1000000.
            action, log_prob, log_prob_mask, entropy, value = self.a2c(state, instruction, mask)
            action = action.cpu().numpy()
            action = self.pe_space[action]
            self.record_resource(action)
        else:
            mask = torch.from_numpy(self.layer_pipe_mask).to(self.device).float()
            action, log_prob, log_prob_mask, entropy, value = self.a2c(state, instruction, mask)
            action = action.cpu().numpy()
            # action = self.layer_pipe_space[action]
            self.record_layer_pipe(action)
            if instruction % self.num_pipe == 0:
                self.layer_pipe_mask = np.zeros((self.num_episodes, self.num_pipe))

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

        advantage = dis_rewards - self.values
        policy_loss = (-1 * self.log_probs * self.log_prob_masks * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()

        value_coeff = 5e-6
        entropy_coeff = 1.
        loss = policy_loss + value_coeff * value_loss - entropy_coeff * self.entropies.sum(dim=0).mean()
        # loss = policy_loss + value_coeff * value_loss
        # print(policy_loss, value_loss,  self.entropies.sum(dim=0))
        # print(self.rewards, loss.mean())

        return loss

        # policy_loss = dis_rewards * (-1 * self.log_probs * self.log_prob_masks).sum(dim=0)
        # return policy_loss.mean()


def init_weights(m):
    if type(m) == nn.LSTMCell:
        nn.init.orthogonal_(m.weight_hh)
        nn.init.orthogonal_(m.weight_ih)
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


class A2CNet(nn.Module):
    def __init__(self, pe_space, band_space, layer_pipe_space, in_size1, in_size2, feat_size=128, num_episodes=None):
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
        self.pe_space = pe_space
        self.band_space = band_space
        self.layer_pipe_space = layer_pipe_space
        self.feat_size = feat_size
        self.num_episodes = num_episodes

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

        self.band_decoder = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, len(band_space)),
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

    def forward(self, state, instruction, mask):
        # if instruction == 0:
        #     num_pipe_score = self.x_decoder(dim_feat)
        #     num_pipe_prob = F.softmax(num_pipe_score, dim=-1)
        #     num_pipe_density = Categorical(num_pipe_prob)
        #     num_pipe_action = num_pipe_density.sample()
        #     num_pipe_log_prob = num_pipe_density.log_prob(num_pipe_action)
        #     action = num_pipe_action
        #     log_prob = num_pipe_log_prob
        #     entropy = num_pipe_density.entropy()
        num_pipe = len(self.layer_pipe_space)
        if 1 <= instruction <= num_pipe:
            feat = self.num_pe_encoder(state)
            h, x = self.lstm(feat, self.lstm_value)
            self.lstm_value = (h, x)
            # h = feat
            num_pe_score = self.num_pe_decoder(h) + mask
            num_pe_prob = F.softmax(num_pe_score, dim=-1)
            num_pe_density = Categorical(num_pe_prob)
            num_pe_action = num_pe_density.sample()
            num_pe_log_prob = num_pe_density.log_prob(num_pe_action)
            action = num_pe_action
            log_prob = num_pe_log_prob

            # band_score = self.band_decoder(h) + mask
            # band_prob = F.softmax(band_score, dim=-1)
            # band_density = Categorical(band_prob)
            # band_action = band_density.sample()
            # band_log_prob = band_density.log_prob(band_action)
            # action = band_action
            # log_prob = band_log_prob
            entropy = num_pe_density.entropy()
        else:
            feat = self.layer_pipe_encoder(state)
            h, x = self.lstm(feat, self.lstm_value)
            self.lstm_value = (h, x)
            # h = feat
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
