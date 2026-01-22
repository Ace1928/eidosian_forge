import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.util import nest
def split_arg_into_blocks(block_dims, block_dims_fn, arg, axis=-1):
    """Split `x` into blocks matching `operators`'s `domain_dimension`.

  Specifically, if we have a blockwise lower-triangular matrix, with block
  sizes along the diagonal `[M_j, M_j] j = 0,1,2..J`,  this method splits `arg`
  on `axis` into `J` tensors, whose shape at `axis` is `M_j`.

  Args:
    block_dims: Iterable of `TensorShapes`.
    block_dims_fn: Callable returning an iterable of `Tensor`s.
    arg: `Tensor`. `arg` is split into `J` tensors.
    axis: Python `Integer` representing the axis to split `arg` on.

  Returns:
    A list of `Tensor`s.
  """
    block_sizes = [dim.value for dim in block_dims]
    if any((d is None for d in block_sizes)):
        block_sizes = block_dims_fn()
    return array_ops.split(arg, block_sizes, axis=axis)