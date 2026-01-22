import gym
import itertools
import minerl
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tqdm
import baselines.common.tf_util as U
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule
import logging
import coloredlogs
def observation_wrapper(obs):
    pov = obs['pov'].astype(np.float32) / 255.0 - 0.5
    return pov