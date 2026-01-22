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
@tf_export('strings.unicode_encode')
@dispatch.add_dispatch_support
def unicode_encode(input, output_encoding, errors='replace', replacement_char=65533, name=None):
    """Encodes each sequence of Unicode code points in `input` into a string.

  `result[i1...iN]` is the string formed by concatenating the Unicode
  codepoints `input[1...iN, :]`, encoded using `output_encoding`.

  Args:
    input: An `N+1` dimensional potentially ragged integer tensor with shape
      `[D1...DN, num_chars]`.
    output_encoding: Unicode encoding that should be used to encode each
      codepoint sequence.  Can be `"UTF-8"`, `"UTF-16-BE"`, or `"UTF-32-BE"`.
    errors: Specifies the response when an invalid codepoint is encountered
      (optional). One of:
            * `'replace'`: Replace invalid codepoint with the
              `replacement_char`. (default)
            * `'ignore'`: Skip invalid codepoints.
            * `'strict'`: Raise an exception for any invalid codepoint.
    replacement_char: The replacement character codepoint to be used in place of
      any invalid input when `errors='replace'`. Any valid unicode codepoint may
      be used. The default value is the default unicode replacement character
      which is 0xFFFD (U+65533).
    name: A name for the operation (optional).

  Returns:
    A `N` dimensional `string` tensor with shape `[D1...DN]`.

  #### Example:

  >>> input = tf.ragged.constant(
  ...     [[71, 246, 246, 100, 110, 105, 103, 104, 116], [128522]])
  >>> print(unicode_encode(input, 'UTF-8'))
  tf.Tensor([b'G\\xc3\\xb6\\xc3\\xb6dnight' b'\\xf0\\x9f\\x98\\x8a'],
            shape=(2,), dtype=string)
  """
    with ops.name_scope(name, 'UnicodeEncode', [input]):
        input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
        if input_tensor.shape.ndims is None:
            raise ValueError('Rank of input_tensor must be statically known.')
        if ragged_tensor.is_ragged(input_tensor):
            if input_tensor.flat_values.shape.ndims > 1:
                return input_tensor.with_flat_values(unicode_encode(input_tensor.flat_values, output_encoding, errors, replacement_char))
            elif input_tensor.ragged_rank > 1:
                return input_tensor.with_values(unicode_encode(input_tensor.values, output_encoding, errors, replacement_char))
            else:
                return gen_string_ops.unicode_encode(input_values=input_tensor.values, input_splits=input_tensor.row_splits, output_encoding=output_encoding, errors=errors, replacement_char=replacement_char)
        elif input_tensor.shape.ndims == 2:
            return unicode_encode(ragged_tensor.RaggedTensor.from_tensor(input_tensor), output_encoding, errors, replacement_char)
        elif input_tensor.shape.ndims > 2:
            flat_input_tensor = array_ops.reshape(input_tensor, array_ops_stack.stack([-1, array_ops.shape(input_tensor)[-1]]))
            flat_output_tensor = unicode_encode(flat_input_tensor, output_encoding, errors, replacement_char)
            return array_ops.reshape(flat_output_tensor, input_tensor.shape[:-1])
        elif input_tensor.shape.ndims == 0:
            raise ValueError("input_tensor's rank must be at least 1.")
        else:
            ragged_input_tensor = ragged_tensor.RaggedTensor.from_row_splits(input_tensor, array_ops_stack.stack([0, array_ops.shape(input_tensor, out_type=dtypes.int32)[0]]), validate=False)
            output_tensor = unicode_encode(ragged_input_tensor, output_encoding, errors, replacement_char)
            return array_ops.reshape(output_tensor, [])