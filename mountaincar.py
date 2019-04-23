import matplotlib.pyplot as plt
import gym
import numpy as np
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datetime import datetime
import glob, os
import time

env = gym.make('MountainCar-v0')
env.reset()
x = -0.5
while x > -0.7:
    state, rew, done, info = env.step(0)
    x = state[0]
    print(state)
    print("rew", rew)
    print("done", done)
    print("info", info)
    #exit(0)
    env.render()

time.sleep(3)

while x < 0:
    state, rew, done, info = env.step(2) # take a random action
    x = state[0]
    print(state)
    env.render()

time.sleep(3)

while x > -1:
    state, rew, done, info = env.step(0)
    x = state[0]
    print(state)
    env.render()

time.sleep(3)

while x < 0.5:
    state, rew, done, info = env.step(2) # take a random action
    x = state[0]
    print(state)
    env.render()

env.close()
