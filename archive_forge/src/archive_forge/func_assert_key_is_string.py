import six
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def assert_key_is_string(key):
    if not isinstance(key, six.string_types):
        raise ValueError('key must be a string. Got: type {}. Given key: {}.'.format(type(key), key))