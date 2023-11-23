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
# model = 'resnet18'
# model = 'googlenet'
# model = 'unet'
# model = 'resnet50'
# model = 'vgg16'
# model = 'mnasnet'
# model = 'resnet50'
model = 'mobilenet_v2'
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

num_pe = 4096
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

# env.reset()
# sol = []
# num_pipe = 3
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



print(sol)
print(env.get_reward(np.array(sol)))

env.reset()
sol = []
num_pipe = 13
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
# for i in [  7561, 6301, 8822, 5041, 7561 ,3780 ,3780 ,3780, 3780, 3780 ,3780 ,3780 ,3780]:
#     sol.append(i)
# for i in [  8,    3 ,   7  , 0  ,  4,    10  ,  2  ,  1   ,11 ,   9  , 12  ,  6 ,   5 ,  12,
#     6  ,  9  ,  0 ,   1 ,  11   , 5  ,  7 ,   2   , 4,    3  ,  8  , 10 ,   3,    6,
#    10 ,  12  ,  2  ,  7 ,   4  ,  1   , 5 ,  11 ,   9 ,   0 ,   8 ,  10  , 12 ,   8,
#     7 ,   6 ,   2  ,  3   , 9  ,  1 ,  11 ,   5 ,   4  ,  0]:
#     sol.append(i)
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
num_pipe = 26
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
num_pipe = 32
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
num_pipe = 52
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

for f in glob.glob("*.m"):
    os.remove(f)
for f in glob.glob("*.csv"):
    os.remove(f)