import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
def safe_exp_std_fn(std_for_zero_param: float, min_std):
    std_shift = tf.math.log(std_for_zero_param - min_std)
    fn = lambda scale: safe_exp(scale + std_shift) + min_std
    assert abs(fn(0) - std_for_zero_param) < 0.001
    return fn