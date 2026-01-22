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
def filter_distributed_callbacks(callbacks_list, model):
    """Filter Callbacks based on the worker context when running multi-worker.

  Args:
    callbacks_list: A list of `Callback` instances.
    model: Keras model instance.

  Returns:
    The list of `Callback` instances that should be run on this worker.
  """
    if not model._in_multi_worker_mode():
        raise ValueError('filter_distributed_callbacks() should only be called when Keras is in multi worker mode.')
    callbacks_list = callbacks_list or []
    if not [c for c in callbacks_list if isinstance(c, callbacks.ModelCheckpoint)]:
        logging.warning('ModelCheckpoint callback is not provided. Workers will need to restart training if any fails.')
    if callbacks_list is None or is_current_worker_chief():
        return callbacks_list
    return [callback for callback in callbacks_list if not callback._chief_worker_only]