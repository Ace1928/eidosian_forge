import typing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat as util_compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def ragged_tensor_to_string(rt, summarize=None):
    """Returns a scalar string tensor with the contents of a RaggedTensor.

  Requires that `rt.shape.rank` is not `None`.

  Note: this converts the entire `RaggedTensor` into a single string scalar.
  If you want to convert individual elements, use `tf.strings.as_string(rt)`.

  >>> rt1 = tf.ragged.constant([[1, 2, 3], [4, 5]])
  >>> ragged_tensor_to_string(rt1).numpy()
  b'[[1, 2, 3], [4, 5]]'

  >>> rt2 = tf.ragged.constant([[['a'], ['b', 'c']], [['d', 'e', 'f'], []]])
  >>> ragged_tensor_to_string(rt2).numpy()
  b"[[['a'], ['b', 'c']], [['d', 'e', 'f'], []]]"

  >>> rt3 = tf.ragged.constant([[1], [2, 3, 4, 5, 6], [], [], [7], [8, 9]])
  >>> ragged_tensor_to_string(rt3, summarize=2).numpy()
  b'[[1], [2, 3, ..., 5, 6], ..., [7], [8, 9]]'

  Args:
    rt: The RaggedTensor that should be converted to a string.
    summarize: If specified, then only the first and last `summarize` elements
      within each dimension are included in the string. If `-1` or `None`, then
      all elements are included.
  """
    if summarize is not None and summarize != -1 and (not (isinstance(summarize, int) and summarize > 0)):
        raise ValueError('Expected summarize to be -1 or a positive int, got %r' % summarize)
    with ops.name_scope(None, 'AsString', [rt]):
        rt = ragged_tensor.convert_to_tensor_or_ragged_tensor(rt)
        if rt.shape.rank is None:
            raise ValueError('RaggedTensor to_string requires that rt.shape.rank is not None.')
        if rt.dtype == dtypes.string:
            escaped = string_ops.regex_replace(rt.flat_values, "(['\\\\])", '\\\\\\1')
            str_t = rt.with_flat_values("'" + escaped + "'")
        else:
            str_t = rt.with_flat_values(string_ops.as_string(rt.flat_values))
        return _ragged_tensor_to_string(str_t, summarize)