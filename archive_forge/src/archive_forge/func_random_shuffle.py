from typing import Sequence
from tensorflow.core.config import flags
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
@dispatch.dispatch_for_types(random_ops.random_shuffle, StructuredTensor)
def random_shuffle(value, seed=None, name=None):
    """Shuffle a structured tensor on the zeroth axis.

  Args:
    value: a structured tensor of rank at least one.
    seed: the seed for shuffling.
    name: the name for shuffle.

  Returns:
    The shuffled structured tensor.
  """
    with ops.name_scope(name, 'shuffle', [value, seed]):
        if value.rank == 0:
            raise ValueError('Cannot shuffle a scalar StructuredTensor')
        first_dimension = value.nrows()
        index = random_ops.random_shuffle(math_ops.range(first_dimension), seed=seed)
        return gather(value, index, axis=0)