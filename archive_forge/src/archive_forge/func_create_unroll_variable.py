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
def create_unroll_variable(spec):
    z = tf.zeros([num_envs, self._full_length] + spec.shape.dims, dtype=spec.dtype)
    return tf.Variable(z, trainable=False, name=spec.name)