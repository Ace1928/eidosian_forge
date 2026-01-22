import collections
import contextlib
import copy
import platform
import random
import threading
import numpy as np
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import backend
from keras.src.engine import keras_tensor
from keras.src.utils import object_identity
from keras.src.utils import tf_contextlib
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python import pywrap_tfe
def validate_axis(axis, input_shape):
    """Validate an axis value and returns its standardized form.

    Args:
      axis: Value to validate. Can be an integer or a list/tuple of integers.
        Integers may be negative.
      input_shape: Reference input shape that the axis/axes refer to.

    Returns:
      Normalized form of `axis`, i.e. a list with all-positive values.
    """
    input_shape = tf.TensorShape(input_shape)
    rank = input_shape.rank
    if not rank:
        raise ValueError(f'Input has undefined rank. Received: input_shape={input_shape}')
    if isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis)
    for idx, x in enumerate(axis):
        if x < 0:
            axis[idx] = rank + x
    for x in axis:
        if x < 0 or x >= rank:
            raise ValueError(f'Invalid value for `axis` argument. Expected 0 <= axis < inputs.rank (with inputs.rank={rank}). Received: axis={tuple(axis)}')
    if len(axis) != len(set(axis)):
        raise ValueError(f'Duplicate axis: {tuple(axis)}')
    return axis