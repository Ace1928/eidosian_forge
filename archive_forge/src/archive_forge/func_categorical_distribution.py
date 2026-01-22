import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
def categorical_distribution(n_actions, dtype):
    """Initialize the categorical distribution.

  Args:
    n_actions: the number of actions available.
    dtype: dtype of actions, usually int32 or int64.

  Returns:
    A tuple (param size, fn(params) -> distribution)
  """

    def create_dist(parameters):
        return tfd.Categorical(logits=parameters, dtype=dtype)
    return ParametricDistribution(n_actions, create_dist)