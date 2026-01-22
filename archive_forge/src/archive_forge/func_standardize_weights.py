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
def standardize_weights(y, sample_weight=None, class_weight=None, sample_weight_mode=None):
    """Performs sample weight validation and standardization.

  Everything gets normalized to a single sample-wise (or timestep-wise)
  weight array. If both `sample_weight` and `class_weight` are provided,
  the weights are multiplied.

  Args:
      y: Numpy array or Tensor of model targets to be weighted.
      sample_weight: User-provided `sample_weight` argument.
      class_weight: User-provided `class_weight` argument.
      sample_weight_mode: One of `None` or `"temporal"`. `"temporal"` indicated
        that we expect 2D weight data that will be applied to the last 2
        dimensions of the targets (i.e. we are weighting timesteps, not
        samples).

  Returns:
      A numpy array of target weights, one entry per sample to weight.

  Raises:
      ValueError: In case of invalid user-provided arguments.
  """
    if isinstance(sample_weight, tuple):
        sample_weight = sample_weight[0]
    if sample_weight_mode is not None and sample_weight_mode != 'samplewise':
        if sample_weight_mode != 'temporal':
            raise ValueError('"sample_weight_mode should be None or "temporal". Found: ' + str(sample_weight_mode))
        if len(y.shape) < 3:
            raise ValueError('Found a sample_weight array for an input with shape ' + str(y.shape) + '. Timestep-wise sample weighting (use of sample_weight_mode="temporal") is restricted to outputs that are at least 3D, i.e. that have a time dimension.')
        if sample_weight is not None and len(sample_weight.shape) != 2:
            raise ValueError('Found a sample_weight array with shape ' + str(sample_weight.shape) + '. In order to use timestep-wise sample weighting, you should pass a 2D sample_weight array.')
    elif sample_weight is not None and len(sample_weight.shape) != 1:
        raise ValueError('Found a sample_weight array with shape {}. In order to use timestep-wise sample weights, you should specify sample_weight_mode="temporal" in compile(); founssd "{}" instead. If you just mean to use sample-wise weights, make sure your sample_weight array is 1D.'.format(sample_weight.shape, sample_weight_mode))
    if sample_weight is not None:
        if len(sample_weight.shape) > len(y.shape):
            raise ValueError('Found a sample_weight with shape' + str(sample_weight.shape) + '.Expected sample_weight with rank less than or equal to ' + str(len(y.shape)))
        if not tensor_util.is_tf_type(sample_weight) and y.shape[:sample_weight.ndim] != sample_weight.shape:
            raise ValueError('Found a sample_weight array with shape ' + str(sample_weight.shape) + ' for an input with shape ' + str(y.shape) + '. sample_weight cannot be broadcast.')
    class_sample_weight = None
    if isinstance(class_weight, dict):
        if len(y.shape) > 2:
            raise ValueError('`class_weight` not supported for 3+ dimensional targets.')
        if tensor_util.is_tf_type(y):
            keys = np.array(sorted(class_weight.keys()))
            values = np.array([class_weight[i] for i in keys])
            weight_vector = np.zeros(np.max(keys) + 1)
            weight_vector[:] = np.nan
            weight_vector[keys] = values
            y_classes = smart_cond.smart_cond(len(y.shape.as_list()) == 2 and backend.shape(y)[1] > 1, lambda: backend.argmax(y, axis=1), lambda: math_ops.cast(backend.reshape(y, (-1,)), dtypes.int64))
            class_sample_weight = array_ops.gather(weight_vector, y_classes)
            gen_array_ops.check_numerics(class_sample_weight, 'Invalid classes or class weights detected. NaN values indicate that an appropriate class weight could not be determined.')
            class_sample_weight = math_ops.cast(class_sample_weight, backend.floatx())
            if sample_weight is not None:
                sample_weight = math_ops.cast(tensor_conversion.convert_to_tensor_v2_with_dispatch(sample_weight), backend.floatx())
        else:
            y_classes = y
            if len(y.shape) == 2:
                if y.shape[1] > 1:
                    y_classes = np.argmax(y, axis=1)
                elif y.shape[1] == 1:
                    y_classes = np.reshape(y, y.shape[0])
            class_sample_weight = np.asarray([class_weight[cls] for cls in y_classes if cls in class_weight])
            if len(class_sample_weight) != len(y_classes):
                existing_classes = set(y_classes)
                existing_class_weight = set(class_weight.keys())
                raise ValueError('`class_weight` must contain all classes in the data. The classes %s exist in the data but not in `class_weight`.' % (existing_classes - existing_class_weight))
    if class_sample_weight is not None and sample_weight is not None:
        return class_sample_weight * sample_weight
    if sample_weight is not None:
        return sample_weight
    if class_sample_weight is not None:
        return class_sample_weight
    return None