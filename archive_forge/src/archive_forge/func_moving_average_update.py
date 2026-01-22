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
@doc_controls.do_not_generate_docs
def moving_average_update(x, value, momentum):
    """Compute the exponential moving average of a value.

  The moving average 'x' is updated with 'value' following:

  ```
  x = x * momentum + value * (1 - momentum)
  ```

  For example:

  >>> x = tf.Variable(0.0)
  >>> momentum=0.9
  >>> moving_average_update(x, value = 2.0, momentum=momentum).numpy()
  >>> x.numpy()
  0.2

  The result will be biased towards the initial value of the variable.

  If the variable was initialized to zero, you can divide by
  `1 - momentum ** num_updates` to debias it (Section 3 of
  [Kingma et al., 2015](https://arxiv.org/abs/1412.6980)):

  >>> num_updates = 1.0
  >>> x_zdb = x/(1 - momentum**num_updates)
  >>> x_zdb.numpy()
  2.0

  Args:
      x: A Variable, the moving average.
      value: A tensor with the same shape as `x`, the new value to be
        averaged in.
      momentum: The moving average momentum.

  Returns:
      The updated variable.
  """
    if tf2.enabled():
        momentum = math_ops.cast(momentum, x.dtype)
        value = math_ops.cast(value, x.dtype)
        return x.assign(x * momentum + value * (1 - momentum))
    else:
        return moving_averages.assign_moving_average(x, value, momentum, zero_debias=True)