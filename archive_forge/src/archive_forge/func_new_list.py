import collections
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
def new_list(iterable=None):
    """The list constructor.

  Args:
    iterable: Optional elements to fill the list with.

  Returns:
    A list-like object. The exact return value depends on the initial elements.
  """
    if iterable:
        elements = tuple(iterable)
    else:
        elements = ()
    if elements:
        return _py_list_new(elements)
    return tf_tensor_list_new(elements)