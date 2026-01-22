import functools
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
def validate_per_replica_inputs(distribution_strategy, x):
    """Validates PerReplica dataset input list.

  Args:
    distribution_strategy: The current DistributionStrategy used to call
      `fit`, `evaluate` and `predict`.
    x: A list of PerReplica objects that represent the input or
      target values.

  Returns:
    List containing the first element of each of the PerReplica objects in
    the input list.

  Raises:
    ValueError: If any of the objects in the `per_replica_list` is not a tensor.

  """
    per_replica_list = nest.flatten(x, expand_composites=True)
    x_values_list = []
    for x in per_replica_list:
        x_values = distribution_strategy.unwrap(x)
        for value in x_values:
            if not tensor_util.is_tf_type(value):
                raise ValueError('Dataset input to the model should be tensors instead they are of type {}'.format(type(value)))
        if not context.executing_eagerly():
            validate_all_tensor_shapes(x, x_values)
        validate_all_tensor_types(x, x_values)
        x_values_list.append(x_values[0])
    return x_values_list