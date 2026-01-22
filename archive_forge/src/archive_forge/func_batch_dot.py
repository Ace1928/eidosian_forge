import collections
import itertools
import json
import os
import sys
import threading
import warnings
import weakref
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session as session_module
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.eager.context import get_config
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradients_module
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import moving_averages
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def batch_dot(x, y, axes=None):
    """Batchwise dot product.

  `batch_dot` is used to compute dot product of `x` and `y` when
  `x` and `y` are data in batch, i.e. in a shape of
  `(batch_size, :)`.
  `batch_dot` results in a tensor or variable with less dimensions
  than the input. If the number of dimensions is reduced to 1,
  we use `expand_dims` to make sure that ndim is at least 2.

  Args:
    x: Keras tensor or variable with `ndim >= 2`.
    y: Keras tensor or variable with `ndim >= 2`.
    axes: Tuple or list of integers with target dimensions, or single integer.
      The sizes of `x.shape[axes[0]]` and `y.shape[axes[1]]` should be equal.

  Returns:
    A tensor with shape equal to the concatenation of `x`'s shape
    (less the dimension that was summed over) and `y`'s shape
    (less the batch dimension and the dimension that was summed over).
    If the final rank is 1, we reshape it to `(batch_size, 1)`.

  Examples:

  >>> x_batch = tf.keras.backend.ones(shape=(32, 20, 1))
  >>> y_batch = tf.keras.backend.ones(shape=(32, 30, 20))
  >>> xy_batch_dot = tf.keras.backend.batch_dot(x_batch, y_batch, axes=(1, 2))
  >>> tf.keras.backend.int_shape(xy_batch_dot)
  (32, 1, 30)

  Shape inference:
    Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
    If `axes` is (1, 2), to find the output shape of resultant tensor,
        loop through each dimension in `x`'s shape and `y`'s shape:
    * `x.shape[0]` : 100 : append to output shape
    * `x.shape[1]` : 20 : do not append to output shape,
        dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
    * `y.shape[0]` : 100 : do not append to output shape,
        always ignore first dimension of `y`
    * `y.shape[1]` : 30 : append to output shape
    * `y.shape[2]` : 20 : do not append to output shape,
        dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
    `output_shape` = `(100, 30)`
  """
    x_shape = int_shape(x)
    y_shape = int_shape(y)
    x_ndim = len(x_shape)
    y_ndim = len(y_shape)
    if x_ndim < 2 or y_ndim < 2:
        raise ValueError('Cannot do batch_dot on inputs with rank < 2. Received inputs with shapes ' + str(x_shape) + ' and ' + str(y_shape) + '.')
    x_batch_size = x_shape[0]
    y_batch_size = y_shape[0]
    if x_batch_size is not None and y_batch_size is not None:
        if x_batch_size != y_batch_size:
            raise ValueError('Cannot do batch_dot on inputs with different batch sizes. Received inputs with shapes ' + str(x_shape) + ' and ' + str(y_shape) + '.')
    if isinstance(axes, int):
        axes = [axes, axes]
    if axes is None:
        if y_ndim == 2:
            axes = [x_ndim - 1, y_ndim - 1]
        else:
            axes = [x_ndim - 1, y_ndim - 2]
    if py_any((isinstance(a, (list, tuple)) for a in axes)):
        raise ValueError('Multiple target dimensions are not supported. ' + 'Expected: None, int, (int, int), ' + 'Provided: ' + str(axes))
    axes = list(axes)
    if axes[0] < 0:
        axes[0] += x_ndim
    if axes[1] < 0:
        axes[1] += y_ndim
    if 0 in axes:
        raise ValueError('Cannot perform batch_dot over axis 0. If your inputs are not batched, add a dummy batch dimension to your inputs using K.expand_dims(x, 0)')
    a0, a1 = axes
    d1 = x_shape[a0]
    d2 = y_shape[a1]
    if d1 is not None and d2 is not None and (d1 != d2):
        raise ValueError('Cannot do batch_dot on inputs with shapes ' + str(x_shape) + ' and ' + str(y_shape) + ' with axes=' + str(axes) + '. x.shape[%d] != y.shape[%d] (%d != %d).' % (axes[0], axes[1], d1, d2))
    orig_x_ndim = x_ndim
    orig_y_ndim = y_ndim
    if x_ndim == 2:
        x = array_ops.expand_dims(x, 1)
        a0 += 1
        x_ndim += 1
    if y_ndim == 2:
        y = array_ops.expand_dims(y, 2)
        y_ndim += 1
    if a0 != x_ndim - 1:
        pattern = list(range(x_ndim))
        for i in range(a0, x_ndim - 1):
            pattern[i] = pattern[i + 1]
        pattern[-1] = a0
        x = array_ops.transpose(x, pattern)
    if a1 != 1:
        pattern = list(range(y_ndim))
        for i in range(a1, 1, -1):
            pattern[i] = pattern[i - 1]
        pattern[1] = a1
        y = array_ops.transpose(y, pattern)
    if x_ndim > 3:
        x_shape = shape(x)
        x_mid_dims = x_shape[1:-1]
        x_squashed_shape = array_ops_stack.stack([x_shape[0], -1, x_shape[-1]])
        x = array_ops.reshape(x, x_squashed_shape)
        x_squashed = True
    else:
        x_squashed = False
    if y_ndim > 3:
        y_shape = shape(y)
        y_trail_dims = y_shape[2:]
        y_squashed_shape = array_ops_stack.stack([y_shape[0], y_shape[1], -1])
        y = array_ops.reshape(y, y_squashed_shape)
        y_squashed = True
    else:
        y_squashed = False
    result = math_ops.matmul(x, y)
    output_shape = array_ops.shape(result)
    do_reshape = False
    if x_squashed:
        output_shape = array_ops.concat([output_shape[:1], x_mid_dims, output_shape[-1:]], 0)
        do_reshape = True
    if y_squashed:
        output_shape = array_ops.concat([output_shape[:-1], y_trail_dims], 0)
        do_reshape = True
    if do_reshape:
        result = array_ops.reshape(result, output_shape)
    if orig_x_ndim == 2:
        result = array_ops.squeeze(result, 1)
    elif orig_y_ndim == 2:
        result = array_ops.squeeze(result, -1)
    return result