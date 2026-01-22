import six
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def check_default_value(shape, default_value, dtype, key):
    """Returns default value as tuple if it's valid, otherwise raises errors.

  This function verifies that `default_value` is compatible with both `shape`
  and `dtype`. If it is not compatible, it raises an error. If it is compatible,
  it casts default_value to a tuple and returns it. `key` is used only
  for error message.

  Args:
    shape: An iterable of integers specifies the shape of the `Tensor`.
    default_value: If a single value is provided, the same value will be applied
      as the default value for every item. If an iterable of values is
      provided, the shape of the `default_value` should be equal to the given
      `shape`.
    dtype: defines the type of values. Default value is `tf.float32`. Must be a
      non-quantized, real integer or floating point type.
    key: Column name, used only for error messages.

  Returns:
    A tuple which will be used as default value.

  Raises:
    TypeError: if `default_value` is an iterable but not compatible with `shape`
    TypeError: if `default_value` is not compatible with `dtype`.
    ValueError: if `dtype` is not convertible to `tf.float32`.
  """
    if default_value is None:
        return None
    if isinstance(default_value, int):
        return _create_tuple(shape, default_value)
    if isinstance(default_value, float) and dtype.is_floating:
        return _create_tuple(shape, default_value)
    if callable(getattr(default_value, 'tolist', None)):
        default_value = default_value.tolist()
    if nest.is_nested(default_value):
        if not _is_shape_and_default_value_compatible(default_value, shape):
            raise ValueError('The shape of default_value must be equal to given shape. default_value: {}, shape: {}, key: {}'.format(default_value, shape, key))
        is_list_all_int = all((isinstance(v, int) for v in nest.flatten(default_value)))
        is_list_has_float = any((isinstance(v, float) for v in nest.flatten(default_value)))
        if is_list_all_int:
            return _as_tuple(default_value)
        if is_list_has_float and dtype.is_floating:
            return _as_tuple(default_value)
    raise TypeError('default_value must be compatible with dtype. default_value: {}, dtype: {}, key: {}'.format(default_value, dtype, key))