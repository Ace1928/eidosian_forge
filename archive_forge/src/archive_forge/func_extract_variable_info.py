import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def extract_variable_info(kwargs) -> Tuple[Text, Tuple[int, ...], dtypes.DType, Callable[[], Any]]:
    """Extracts the variable creation attributes from the kwargs.

  Args:
    kwargs: a dict of keyword arguments that were passed to a variable creator
      scope.

  Returns:
    A tuple of variable name, shape, dtype, initialization function.
  """
    if isinstance(kwargs['initial_value'], functools.partial) and ('shape' in kwargs['initial_value'].keywords or kwargs['initial_value'].args):
        if 'shape' in kwargs['initial_value'].keywords:
            shape = kwargs['initial_value'].keywords['shape']
        else:
            shape = kwargs['initial_value'].args[0]
        return (kwargs['name'], shape, kwargs['initial_value'].keywords.get('dtype', kwargs['dtype']), kwargs['initial_value'].func)
    elif 'shape' not in kwargs or kwargs['shape'] is None or (not callable(kwargs['initial_value'])):
        raise ValueError('Unable to extract initializer function and shape from {}. Please either pass a function that expects a shape and dtype as the initial value for your variable or functools.partial object with the shape and dtype kwargs set. This is needed so that we can initialize the shards of the ShardedVariable locally.'.format(kwargs['initial_value']))
    else:
        return (kwargs['name'], kwargs['shape'], kwargs['dtype'], kwargs['initial_value'])