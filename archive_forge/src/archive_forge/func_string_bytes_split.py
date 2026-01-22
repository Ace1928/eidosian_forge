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
@tf_export('strings.bytes_split')
@dispatch.add_dispatch_support
def string_bytes_split(input, name=None):
    """Split string elements of `input` into bytes.

  Examples:

  >>> tf.strings.bytes_split('hello').numpy()
  array([b'h', b'e', b'l', b'l', b'o'], dtype=object)
  >>> tf.strings.bytes_split(['hello', '123'])
  <tf.RaggedTensor [[b'h', b'e', b'l', b'l', b'o'], [b'1', b'2', b'3']]>

  Note that this op splits strings into bytes, not unicode characters.  To
  split strings into unicode characters, use `tf.strings.unicode_split`.

  See also: `tf.io.decode_raw`, `tf.strings.split`, `tf.strings.unicode_split`.

  Args:
    input: A string `Tensor` or `RaggedTensor`: the strings to split.  Must
      have a statically known rank (`N`).
    name: A name for the operation (optional).

  Returns:
    A `RaggedTensor` of rank `N+1`: the bytes that make up the source strings.
  """
    with ops.name_scope(name, 'StringsByteSplit', [input]):
        input = ragged_tensor.convert_to_tensor_or_ragged_tensor(input, name='input')
        if isinstance(input, ragged_tensor.RaggedTensor):
            return input.with_flat_values(string_bytes_split(input.flat_values))
        rank = input.shape.ndims
        if rank is None:
            raise ValueError('input must have a statically-known rank.')
        if rank == 0:
            return string_bytes_split(array_ops_stack.stack([input]))[0]
        elif rank == 1:
            indices, values, shape = gen_string_ops.string_split(input, delimiter='', skip_empty=False)
            return ragged_tensor.RaggedTensor.from_value_rowids(values=values, value_rowids=indices[:, 0], nrows=shape[0], validate=False)
        else:
            return string_bytes_split(ragged_tensor.RaggedTensor.from_tensor(input))