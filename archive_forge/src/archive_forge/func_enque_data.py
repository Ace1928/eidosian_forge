import time
import zmq
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, LSTMCell
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import threading
import random
import collections
import argparse
from absl import flags
from absl import logging
from typing import Any, List, Sequence, Tuple
from gym.spaces import Dict, Discrete, Box, Tuple
import network
from parametric_distribution import get_parametric_distribution_for_action_space
@tf.function
def enque_data(env_ids, rewards, dones, states, policies, actions, memory_states, carry_states):
    queue.enqueue((env_ids, rewards, dones, states, policies, actions, memory_states, carry_states))