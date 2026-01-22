from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_getitem
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_decorator
def ragged_hash(self):
    """The operation invoked by the `RaggedTensor.__hash__` operator."""
    g = getattr(self.row_splits, 'graph', None)
    if tensor.Tensor._USE_EQUALITY and ops.executing_eagerly_outside_functions() and (g is None or g.building_function):
        raise TypeError('RaggedTensor is unhashable.')
    else:
        return id(self)