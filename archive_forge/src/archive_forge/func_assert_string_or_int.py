import six
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def assert_string_or_int(dtype, prefix):
    if dtype != dtypes.string and (not dtype.is_integer):
        raise ValueError('{} dtype must be string or integer. dtype: {}.'.format(prefix, dtype))