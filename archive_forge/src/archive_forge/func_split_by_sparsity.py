import copy
import threading
from typing import Callable, List, Optional, Union
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
def split_by_sparsity(values):
    """Split values into dense and sparse values.

  Args:
    values: a list of tensors or `PerReplica`s.

  Returns:
    Four lists:
      a list of dense values, a list of their indices in `values` and
      a list of sparse values, a list of their indices in `values`.
  """
    dense_values = []
    dense_indices = []
    sparse_values = []
    sparse_indices = []
    for i, v in enumerate(values):
        if is_indexed_slices(v):
            sparse_values.append(v)
            sparse_indices.append(i)
        else:
            dense_values.append(v)
            dense_indices.append(i)
    return (dense_values, dense_indices, sparse_values, sparse_indices)