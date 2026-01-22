import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
def softplus_default_std_fn(scale):
    return tf.nn.softplus(scale) + 0.001