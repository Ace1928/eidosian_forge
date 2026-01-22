import abc
import atexit
import collections
import functools
import multiprocessing.pool
import threading
import time
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def get_loss_function(loss):
    """Returns the loss corresponding to the loss input in `compile` API."""
    if loss is None or isinstance(loss, losses.Loss):
        return loss
    if tf_inspect.isclass(loss) and issubclass(loss, losses.Loss):
        raise ValueError('Received uninstantiated Loss class: {}\nPlease call loss ""classes before passing them to Model.compile.'.format(loss))
    if isinstance(loss, collections.abc.Mapping):
        loss = losses.get(loss)
    if callable(loss) and (not hasattr(loss, '__name__')):
        return loss
    loss_fn = losses.get(loss)
    return losses.LossFunctionWrapper(loss_fn, name=loss_fn.__name__, reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE)