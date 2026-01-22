import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
def normal_clipped_distribution(num_actions, gaussian_std_fn=softplus_default_std_fn):
    """Normal distribution postprocessed by a clipped identity."""

    def create_dist(parameters):
        loc, scale = tf.split(parameters, 2, axis=-1)
        scale = gaussian_std_fn(scale)
        normal_dist = tfd.Normal(loc=loc, scale=scale)
        return tfd.Independent(CLIPPED_IDENTITY(normal_dist), reinterpreted_batch_ndims=1)
    return ParametricDistribution(2 * num_actions, create_dist)