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
def pack_x_y_sample_weight(x, y=None, sample_weight=None):
    """Packs user-provided data into a tuple.

  This is a convenience utility for packing data into the tuple formats
  that `Model.fit` uses.

  Standalone usage:

  >>> x = tf.ones((10, 1))
  >>> data = tf.keras.utils.pack_x_y_sample_weight(x)
  >>> isinstance(data, tf.Tensor)
  True
  >>> y = tf.ones((10, 1))
  >>> data = tf.keras.utils.pack_x_y_sample_weight(x, y)
  >>> isinstance(data, tuple)
  True
  >>> x, y = data

  Args:
    x: Features to pass to `Model`.
    y: Ground-truth targets to pass to `Model`.
    sample_weight: Sample weight for each element.

  Returns:
    Tuple in the format used in `Model.fit`.
  """
    if y is None:
        if not nest.is_nested(x):
            return x
        else:
            return (x,)
    elif sample_weight is None:
        return (x, y)
    else:
        return (x, y, sample_weight)