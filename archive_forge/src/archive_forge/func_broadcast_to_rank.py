from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
def broadcast_to_rank(self, rank):
    """Adds leading size-1 dimensions to broadcast `self` to the given rank.

    E.g., if `shape1` is `[3, (D2), 4]`, then `shape1.broadcast_to_rank(5)`
    is `[1, 1, 3, (D2), 4]`.

    Args:
      rank: The rank for the returned shape.

    Returns:
      A RaggedTensorDynamicShape with `rank` dimensions, whose inner dimensions
      have the same size as `self` and whose outer dimensions have size `1`.

    Raises:
      ValueError: If `self.rank` is unknown or greater than `rank`.
    """
    if self.rank is None:
        raise ValueError('Unable to broadcast: self.rank is unknown')
    dims_to_add = rank - self.rank
    if dims_to_add < 0:
        raise ValueError('Unable to broadcast: rank=%d must be greater than self.rank=%d.' % (rank, self.rank))
    elif dims_to_add == 0:
        return self
    elif self._partitioned_dim_sizes:
        partitioned_dims = (1,) * dims_to_add + self._partitioned_dim_sizes
        return RaggedTensorDynamicShape(partitioned_dims, self.inner_dim_sizes, self.dim_size_dtype)
    else:
        inner_dims = array_ops.concat([array_ops.ones([dims_to_add], self.dim_size_dtype), self.inner_dim_sizes], axis=0)
        return RaggedTensorDynamicShape([], inner_dims, self.dim_size_dtype)