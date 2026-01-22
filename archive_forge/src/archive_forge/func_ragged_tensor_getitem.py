from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('__operators__.ragged_getitem', v1=[])
@dispatch.add_dispatch_support
def ragged_tensor_getitem(rt_input, key):
    """Returns the specified piece of this RaggedTensor.

  Supports multidimensional indexing and slicing, with one restriction:
  indexing into a ragged inner dimension is not allowed.  This case is
  problematic because the indicated value may exist in some rows but not
  others.  In such cases, it's not obvious whether we should (1) report an
  IndexError; (2) use a default value; or (3) skip that value and return a
  tensor with fewer rows than we started with.  Following the guiding
  principles of Python ("In the face of ambiguity, refuse the temptation to
  guess"), we simply disallow this operation.

  Args:
    rt_input: The RaggedTensor to slice.
    key: Indicates which piece of the RaggedTensor to return, using standard
      Python semantics (e.g., negative values index from the end).  `key`
      may have any of the following types:

      * `int` constant
      * Scalar integer `Tensor`
      * `slice` containing integer constants and/or scalar integer
        `Tensor`s
      * `Ellipsis`
      * `tf.newaxis`
      * `tuple` containing any of the above (for multidimensional indexing)

  Returns:
    A `Tensor` or `RaggedTensor` object.  Values that include at least one
    ragged dimension are returned as `RaggedTensor`.  Values that include no
    ragged dimensions are returned as `Tensor`.  See above for examples of
    expressions that return `Tensor`s vs `RaggedTensor`s.

  Raises:
    ValueError: If `key` is out of bounds.
    ValueError: If `key` is not supported.
    TypeError: If the indices in `key` have an unsupported type.

  Examples:

  >>> # A 2-D ragged tensor with 1 ragged dimension.
  >>> rt = tf.ragged.constant([['a', 'b', 'c'], ['d', 'e'], ['f'], ['g']])
  >>> rt[0].numpy()                 # First row (1-D `Tensor`)
  array([b'a', b'b', b'c'], dtype=object)
  >>> rt[:3].to_list()              # First three rows (2-D RaggedTensor)
  [[b'a', b'b', b'c'], [b'd', b'e'], [b'f']]
  >>> rt[3, 0].numpy()              # 1st element of 4th row (scalar)
  b'g'

  >>> # A 3-D ragged tensor with 2 ragged dimensions.
  >>> rt = tf.ragged.constant([[[1, 2, 3], [4]],
  ...                          [[5], [], [6]],
  ...                          [[7]],
  ...                          [[8, 9], [10]]])
  >>> rt[1].to_list()               # Second row (2-D RaggedTensor)
  [[5], [], [6]]
  >>> rt[3, 0].numpy()              # First element of fourth row (1-D Tensor)
  array([8, 9], dtype=int32)
  >>> rt[:, 1:3].to_list()          # Items 1-3 of each row (3-D RaggedTensor)
  [[[4]], [[], [6]], [], [[10]]]
  >>> rt[:, -1:].to_list()          # Last item of each row (3-D RaggedTensor)
  [[[4]], [[6]], [[7]], [[10]]]
  """
    if not isinstance(rt_input, ragged_tensor.RaggedTensor):
        raise TypeError('Ragged __getitem__ expects a ragged_tensor.')
    scope_tensors = [rt_input] + list(_tensors_in_key_list(key))
    if isinstance(key, (list, tuple)):
        key = list(key)
    else:
        key = [key]
    with ops.name_scope(None, 'RaggedGetItem', scope_tensors):
        return _ragged_getitem(rt_input, key)