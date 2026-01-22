from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
def ragged_to_dense(rt_input, default_value=None, shape=None):
    """Create a dense tensor from a ragged tensor."""
    return rt_input.to_tensor(default_value=default_value, shape=shape)