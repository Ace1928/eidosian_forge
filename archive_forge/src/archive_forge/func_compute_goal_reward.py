import collections
import pickle
import threading
import time
import timeit
from absl import flags
from absl import logging
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.distribute import values as values_lib  
from tensorflow.python.framework import composite_tensor  
from tensorflow.python.framework import tensor_conversion_registry  
def compute_goal_reward():
    goal_reward = self._compute_reward_fn(achieved_goal=observation['achieved_goal'][:, 1:], desired_goal=observation['desired_goal'][:, :-1])
    return tf.concat(values=[goal_reward[:, :1] * np.nan, goal_reward], axis=1)