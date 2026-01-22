import abc
import contextlib
import functools
import itertools
import math
import random
import numpy as np
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import dataset_creator
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest
def train_validation_split(arrays, validation_split):
    """Split arrays into train and validation subsets in deterministic order.

  The last part of data will become validation data.

  Args:
    arrays: Tensors to split. Allowed inputs are arbitrarily nested structures
      of Tensors and NumPy arrays.
    validation_split: Float between 0 and 1. The proportion of the dataset to
      include in the validation split. The rest of the dataset will be included
      in the training split.
  Returns:
    `(train_arrays, validation_arrays)`
  """

    def _can_split(t):
        tensor_types = _get_tensor_types()
        return isinstance(t, tensor_types) or t is None
    flat_arrays = nest.flatten(arrays)
    unsplitable = [type(t) for t in flat_arrays if not _can_split(t)]
    if unsplitable:
        raise ValueError('`validation_split` is only supported for Tensors or NumPy arrays, found following types in the input: {}'.format(unsplitable))
    if all((t is None for t in flat_arrays)):
        return (arrays, arrays)
    first_non_none = None
    for t in flat_arrays:
        if t is not None:
            first_non_none = t
            break
    batch_dim = int(first_non_none.shape[0])
    split_at = int(math.floor(batch_dim * (1.0 - validation_split)))
    if split_at == 0 or split_at == batch_dim:
        raise ValueError('Training data contains {batch_dim} samples, which is not sufficient to split it into a validation and training set as specified by `validation_split={validation_split}`. Either provide more data, or a different value for the `validation_split` argument.'.format(batch_dim=batch_dim, validation_split=validation_split))

    def _split(t, start, end):
        if t is None:
            return t
        return t[start:end]
    train_arrays = nest.map_structure(functools.partial(_split, start=0, end=split_at), arrays)
    val_arrays = nest.map_structure(functools.partial(_split, start=split_at, end=batch_dim), arrays)
    return (train_arrays, val_arrays)