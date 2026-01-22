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
def split_training_and_validation_data(x, y, sample_weights, validation_split):
    """Split input data into train/eval section based on validation_split."""
    if has_symbolic_tensors(x):
        raise ValueError('If your data is in the form of symbolic tensors, you cannot use `validation_split`.')
    if hasattr(x[0], 'shape'):
        split_at = int(x[0].shape[0] * (1.0 - validation_split))
    else:
        split_at = int(len(x[0]) * (1.0 - validation_split))
    x, val_x = (generic_utils.slice_arrays(x, 0, split_at), generic_utils.slice_arrays(x, split_at))
    y, val_y = (generic_utils.slice_arrays(y, 0, split_at), generic_utils.slice_arrays(y, split_at))
    if sample_weights:
        sample_weights, val_sample_weights = (generic_utils.slice_arrays(sample_weights, 0, split_at), generic_utils.slice_arrays(sample_weights, split_at))
    else:
        val_sample_weights = None
    return (x, y, sample_weights, val_x, val_y, val_sample_weights)