from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_stateful_random_ops
from tensorflow.python.ops import gen_stateless_random_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def uniform_full_int(self, shape, dtype=dtypes.uint64, name=None):
    """Uniform distribution on an integer type's entire range.

    This method is the same as setting `minval` and `maxval` to `None` in the
    `uniform` method.

    Args:
      shape: the shape of the output.
      dtype: (optional) the integer type, default to uint64.
      name: (optional) the name of the node.

    Returns:
      A tensor of random numbers of the required shape.
    """
    dtype = dtypes.as_dtype(dtype)
    with ops.name_scope(name, 'stateful_uniform_full_int', [shape]) as name:
        shape = _shape_tensor(shape)
        return self._uniform_full_int(shape=shape, dtype=dtype, name=name)