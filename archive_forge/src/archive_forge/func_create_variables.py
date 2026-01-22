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
def create_variables(table):
    """Create all variables."""
    variable_shape = (table.vocabulary_size, table.dim)

    def getter(name, shape, dtype, initializer, trainable):
        del shape
        initial_value = functools.partial(initializer, variable_shape, dtype=dtype)
        return tf_variables.Variable(name=name, initial_value=initial_value, shape=variable_shape, dtype=dtype, trainable=trainable)

    def variable_creator(name, initializer, trainable=True):
        return self._add_variable_with_custom_getter(name=name, initializer=initializer, shape=variable_shape, dtype=dtypes.float32, getter=getter, trainable=trainable)
    parameters = variable_creator(table.name, table.initializer, trainable=not self._using_tpu)

    def slot_creator(name, initializer):
        return variable_creator(table.name + '/' + name, initializer, False)
    if table.optimizer is not None:
        slot_vars = table.optimizer._create_slots(parameters, slot_creator)
    else:
        slot_vars = {}
    slot_vars['parameters'] = parameters
    return slot_vars