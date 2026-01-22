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
def tpu_decode(ts, structure=None):
    """Decodes a nest of Tensors encoded with tpu_encode.

  Args:
    ts: A nest of Tensors or TPUEncodedUInt8 composite tensors.
    structure: If not None, a nest of Tensors or TPUEncodedUInt8 composite
      tensors (possibly within PerReplica's) that are only used to recreate the
      structure of `ts` which then should be a list without composite tensors.

  Returns:
    A nest of decoded tensors packed as `structure` if available, otherwise
    packed as `ts`.
  """

    def visit(t, s):
        s = s.values[0] if isinstance(s, values_lib.PerReplica) else s
        if isinstance(s, TPUEncodedUInt8):
            x = t.encoded if isinstance(t, TPUEncodedUInt8) else t
            x = tf.reshape(x, [-1, 32, 1])
            x = tf.broadcast_to(x, x.shape[:-1] + [4])
            x = tf.reshape(x, [-1, 128])
            x = tf.bitwise.bitwise_and(x, [255, 65280, 16711680, 4278190080] * 32)
            x = tf.bitwise.right_shift(x, [0, 8, 16, 24] * 32)
            rank = s.original_shape.rank
            perm = [rank - 1] + list(range(rank - 1))
            inverted_shape = np.array(s.original_shape)[np.argsort(perm)]
            x = tf.reshape(x, inverted_shape)
            x = tf.transpose(x, perm)
            return x
        elif isinstance(s, TPUEncodedF32):
            x = t.encoded if isinstance(t, TPUEncodedF32) else t
            x = tf.reshape(x, s.original_shape)
            return x
        else:
            return t
    return tf.nest.map_structure(visit, ts, structure or ts)