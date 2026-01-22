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
def unwrap_outputs(distribution_strategy, grouped_outputs, with_loss_tensor=False):
    """Unwrap the list of outputs contained in the PerReplica parameters.

  This function calls `flatten_per_replica_values` to parse each of the input
  parameters into a list of outputs on the different devices. If we set
  `with_loss_tensor` to be True, we also call `reduce` on the list of losses on
  the different devices to give us one loss tensor.

  Args:
    distribution_strategy: DistributionStrategy used to distribute training and
        validation.
    grouped_outputs: PerReplica outputs returned from the train or test function
        that we ran on each device.
    with_loss_tensor: Boolean that indicates if we need to add the reduced loss
        tensor as one of the outputs.

  Returns:
    Values of each of the PerReplica outputs.

  """
    if not with_loss_tensor:
        return flatten_per_replica_values(distribution_strategy, grouped_outputs)
    if not isinstance(grouped_outputs, list):
        grouped_outputs = [grouped_outputs]
    loss = distribution_strategy.reduce(reduce_util.ReduceOp.SUM, grouped_outputs[0], axis=None)
    all_outputs = flatten_per_replica_values(distribution_strategy, grouped_outputs[1:])
    if backend.is_tpu_strategy(distribution_strategy) and ops.executing_eagerly_outside_functions():
        all_outputs = all_outputs[::distribution_strategy.num_replicas_in_sync]
    return [loss] + all_outputs