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
def check_steps_argument(input_data, steps, steps_name):
    """Validates `steps` argument based on input data's type.

  The cases when `steps` value must be provided are when
    1. input data passed is an iterator.
    2. model was built on top of symbolic tensors, input data is not
       required and is `None`.
    3. input data passed is a symbolic tensor.

  Args:
      input_data: Input data. Can be Numpy array(s) or TensorFlow tensor(s) or
        tf.data.Dataset iterator or `None`.
      steps: Integer or `None`. Total number of steps (batches of samples) to
        execute.
      steps_name: The public API's parameter name for `steps`.

  Returns:
    boolean, True if `steps` argument is required, else False.

  Raises:
      ValueError: if `steps` argument is required for given input data type
        but not provided.
  """
    is_x_iterator = isinstance(input_data, (iterator_ops.Iterator, iterator_ops.IteratorBase))
    if input_data is None or is_x_iterator or has_symbolic_tensors(input_data) or (isinstance(input_data, list) and (not input_data)):
        if steps is None:
            input_type_str = 'a Dataset iterator' if is_x_iterator else 'data tensors'
            raise ValueError('When using {input_type} as input to a model, you should specify the `{steps_name}` argument.'.format(input_type=input_type_str, steps_name=steps_name))
        return True
    if isinstance(input_data, (data_types.DatasetV1, data_types.DatasetV2)):
        return True
    if steps is not None:
        list_types = (np.ndarray, list, tuple)
        if isinstance(input_data, list_types) or (isinstance(input_data, dict) and any((isinstance(v, list_types) for v in input_data.values()))):
            logging.warning('When passing input data as arrays, do not specify `steps_per_epoch`/`steps` argument. Please use `batch_size` instead.')
    return False