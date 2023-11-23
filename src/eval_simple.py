import math
import os
import random

import numpy as np
from numpy import array, int32
import pandas as pd
import glob
import pickle
import copy

from environment import MaestroEnvironment


# model = 'transformer'
# model = 'shufflenet_v2'
model = 'resnet18'
# model = 'googlenet'
# model = 'unet'
# model = 'resnet50'
# model = 'vgg16'
# model = 'mnasnet'
# model = 'resnet50'
# model = 'mobilenet_v2'
fitness = 'latency'
# fitness = 'energy'
# fitness = 'LEP'
cstr = 'buffer_size'
# dataflow = 'dla'
# dataflow = 'eye'
dataflow = 'shi'
ratio = 0.5
m_file_path = "../data/model/"
m_file = os.path.join(m_file_path, model + ".csv")
df = pd.read_csv(m_file)
model_defs = df.to_numpy()
print(len(model_defs))
_ ,dim_size = model_defs.shape

num_pe = 65536
input_size = 64
bandwidth = 4096

env = MaestroEnvironment(model_defs=model_defs, input_size=input_size, dataflow="dla",
                             num_pe=num_pe, bandwidth=bandwidth,
                             max_pipe=16, fitness='latency', num_episodes=10)

env.reset()
sol = []
num_pipe = 1
sol.append(num_pipe)
for i in range(num_pipe):
    sol.append(int(num_pe / num_pipe))

num_layers = model_defs.shape[0]
groups = num_layers // num_pipe + 1
for g in range(groups):
    for i in range(num_pipe):
        if g*num_pipe + i >= num_layers:
            break
        sol.append(i)
print(sol)
print(env.get_reward(np.array(sol)))

env.reset()
sol = []
num_pipe = 2
sol.append(num_pipe)
for i in range(num_pipe):
    sol.append(int(num_pe / num_pipe))

num_layers = model_defs.shape[0]
groups = num_layers // num_pipe + 1
for g in range(groups):
    for i in range(num_pipe):
        if g*num_pipe + i >= num_layers:
            break
        sol.append(i)
print(sol)
print(env.get_reward(np.array(sol)))

env.reset()
sol = []
num_pipe = 3
sol.append(num_pipe)
for i in range(num_pipe):
    sol.append(int(num_pe / num_pipe))

num_layers = model_defs.shape[0]
groups = num_layers // num_pipe + 1
for g in range(groups):
    for i in range(num_pipe):
        if g*num_pipe + i >= num_layers:
            break
        sol.append(i)
print(sol)
print(env.get_reward(np.array(sol)))

env.reset()
sol = []
num_pipe = 4
sol.append(num_pipe)
for i in range(num_pipe):
    sol.append(int(num_pe / num_pipe))

num_layers = model_defs.shape[0]
groups = num_layers // num_pipe + 1
for g in range(groups):
    for i in range(num_pipe):
        if g*num_pipe + i >= num_layers:
            break
        sol.append(i)
print(sol)
print(env.get_reward(np.array(sol)))

env.reset()
sol = []
num_pipe = 6
sol.append(num_pipe)
for i in range(num_pipe):
    sol.append(int(num_pe / num_pipe))

num_layers = model_defs.shape[0]
groups = num_layers // num_pipe + 1
for g in range(groups):
    for i in range(num_pipe):
        if g*num_pipe + i >= num_layers:
            break
        sol.append(i)
print(sol)
print(env.get_reward(np.array(sol)))

env.reset()
sol = []
num_pipe = 8
sol.append(num_pipe)
for i in range(num_pipe):
    sol.append(int(num_pe / num_pipe))

num_layers = model_defs.shape[0]
groups = num_layers // num_pipe + 1
for g in range(groups):
    for i in range(num_pipe):
        if g*num_pipe + i >= num_layers:
            break
        sol.append(i)

# for i in [ 6144, 16384,  8192,  8192,  6144,  6144,  8192,  6144]:
#     sol.append(i)
# for i in [ 1,     5 ,    0,
#      7  ,   2  ,   6 ,    3 ,    4  ,   7  ,   3 ,    2 ,    5 ,    1 ,    0 ,    4,
#      6  ,   1 ,    3]:
#     sol.append(i)

print(sol)
print(env.get_reward(np.array(sol)))

env.reset()
sol = []
num_pipe = 9
sol.append(num_pipe)
for i in range(num_pipe):
    sol.append(int(num_pe / num_pipe))

num_layers = model_defs.shape[0]
groups = num_layers // num_pipe + 1
for g in range(groups):
    for i in range(num_pipe):
        if g*num_pipe + i >= num_layers:
            break
        sol.append(i)
print(sol)
print(env.get_reward(np.array(sol)))

env.reset()
sol = []
num_pipe = 16
sol.append(num_pipe)
for i in range(num_pipe):
    sol.append(int(num_pe / num_pipe))

num_layers = model_defs.shape[0]
groups = num_layers // num_pipe + 1
for g in range(groups):
    for i in range(num_pipe):
        if g*num_pipe + i >= num_layers:
            break
        sol.append(i)
print(sol)
print(env.get_reward(np.array(sol)))

env.reset()
sol = []
num_pipe = 18
sol.append(num_pipe)
for i in range(num_pipe):
    sol.append(int(num_pe / num_pipe))

num_layers = model_defs.shape[0]
groups = num_layers // num_pipe + 1
for g in range(groups):
    for i in range(num_pipe):
        if g*num_pipe + i >= num_layers:
            break
        sol.append(i)
print(sol)
print(env.get_reward(np.array(sol)))

# env.reset()
# sol = []
# num_pipe = 13
# sol.append(num_pipe)
# for i in range(num_pipe):
#     sol.append(int(num_pe / num_pipe))
#
# num_layers = model_defs.shape[0]
# groups = num_layers // num_pipe + 1
# for g in range(groups):
#     for i in range(num_pipe):
#         if g*num_pipe + i >= num_layers:
#             break
#         sol.append(i)
# print(sol)
# print(env.get_reward(np.array(sol)))
#
# env.reset()
# sol = []
# num_pipe = 16
# sol.append(num_pipe)
# for i in range(num_pipe):
#     sol.append(int(num_pe / num_pipe))
#
# num_layers = model_defs.shape[0]
# groups = num_layers // num_pipe + 1
# for g in range(groups):
#     for i in range(num_pipe):
#         if g*num_pipe + i >= num_layers:
#             break
#         sol.append(i)
# print(sol)
# print(env.get_reward(np.array(sol)))
#
#
# env.reset()
# sol = []
# num_pipe = 25
# sol.append(num_pipe)
# for i in range(num_pipe):
#     sol.append(int(num_pe / num_pipe))
#
# num_layers = model_defs.shape[0]
# groups = num_layers // num_pipe + 1
# for g in range(groups):
#     for i in range(num_pipe):
#         if g*num_pipe + i >= num_layers:
#             break
#         sol.append(i)
# print(sol)
# print(env.get_reward(np.array(sol)))
# #
# env.reset()
# sol = []
# num_pipe = 32
# sol.append(num_pipe)
# for i in range(num_pipe):
#     sol.append(int(num_pe / num_pipe))
#
# num_layers = model_defs.shape[0]
# groups = num_layers // num_pipe + 1
# for g in range(groups):
#     for i in range(num_pipe):
#         if g*num_pipe + i >= num_layers:
#             break
#         sol.append(i)
# print(sol)
# print(env.get_reward(np.array(sol)))
#
#
# env.reset()
# sol = []
# num_pipe = 50
# sol.append(num_pipe)
# for i in range(num_pipe):
#     sol.append(int(num_pe / num_pipe))
#
# num_layers = model_defs.shape[0]
# groups = num_layers // num_pipe + 1
# for g in range(groups):
#     for i in range(num_pipe):
#         if g*num_pipe + i >= num_layers:
#             break
#         sol.append(i)
# print(sol)
# print(env.get_reward(np.array(sol)))
#
# for f in glob.glob("*.m"):
#     os.remove(f)
# for f in glob.glob("*.csv"):
#     os.remove(f)