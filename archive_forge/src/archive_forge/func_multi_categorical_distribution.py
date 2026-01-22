import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
def multi_categorical_distribution(n_dimensions, n_actions_per_dim, dtype):
    """Initialize the categorical distribution.

  Args:
    n_dimensions: the dimensionality of actions.
    n_actions_per_dim: number of actions available per dimension.
    dtype: dtype of actions, usually int32 or int64.

  Returns:
    A tuple (param size, fn(params) -> distribution)
  """

    def create_dist(parameters):
        batch_shape = parameters.shape[:-1]
        logits_shape = [n_dimensions, n_actions_per_dim]
        logits = tf.reshape(parameters, batch_shape + logits_shape)
        return tfd.Independent(tfd.Categorical(logits=logits, dtype=dtype), reinterpreted_batch_ndims=1)
    return ParametricDistribution(n_dimensions * n_actions_per_dim, create_dist)