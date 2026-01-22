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
def group_by_size(input_tensors, bytes_per_pack):
    """Groups `input_tensors` into chunks of `bytes_per_pack`.

  The method preserves the original order of `input_tensors`. The grouping is
  best effort, each pack could have more or less bytes than `bytes_per_pack`.
  It only groups values with known shape.

  Args:
    input_tensors: a list of Tensor.
    bytes_per_pack: an integer.

  Returns:
    A list of packs of Tensor. All values are grouped into one pack if
    `bytes_per_pack` is zero or any of the value has unknown shape.
  """
    if bytes_per_pack == 0:
        return [input_tensors]
    packs = []
    last_pack_size = 0
    for value in input_tensors:
        num_elements = value.shape.num_elements()
        if num_elements is None:
            logging.warning('not packing values due to the unknown or inconsistent shape of %s', value)
            return [input_tensors]
        size = num_elements * value.dtype.size
        if not packs or last_pack_size > bytes_per_pack:
            packs.append([])
            last_pack_size = 0
        packs[-1].append(value)
        last_pack_size += size
    return packs