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
def init_learner(num_training_tpus):
    """Performs common learner initialization."""
    settings = init_learner_multi_host(num_training_tpus)
    if len(settings.hosts) != 1:
        raise ValueError(f'Invalid number of hosts: {len(settings.hosts)}')
    return Settings(settings.strategy, settings.hosts[0][1], settings.training_strategy, settings.encode, settings.decode)