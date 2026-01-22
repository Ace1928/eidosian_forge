import numpy as onp
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export
@tf_export.tf_export('experimental.numpy.random.seed', v1=[])
@np_utils.np_doc('random.seed')
def seed(s):
    """Sets the seed for the random number generator.

  Uses `tf.set_random_seed`.

  Args:
    s: an integer.
  """
    try:
        s = int(s)
    except TypeError:
        raise ValueError(f'Argument `s` got an invalid value {s}. Only integers are supported.')
    random_seed.set_seed(s)