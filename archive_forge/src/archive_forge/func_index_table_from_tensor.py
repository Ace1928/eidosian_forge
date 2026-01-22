import collections
import functools
import uuid
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.gen_lookup_ops import *
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base as trackable_base
from tensorflow.python.trackable import resource
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.util import compat as compat_util
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
def index_table_from_tensor(vocabulary_list, num_oov_buckets=0, default_value=-1, hasher_spec=FastHashSpec, dtype=dtypes.string, name=None):
    """Returns a lookup table that converts a string tensor into int64 IDs.

  This operation constructs a lookup table to convert tensor of strings into
  int64 IDs. The mapping can be initialized from a string `vocabulary_list` 1-D
  tensor where each element is a key and corresponding index within the tensor
  is the value.

  Any lookup of an out-of-vocabulary token will return a bucket ID based on its
  hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the
  `default_value`. The bucket ID range is
  `[vocabulary list size, vocabulary list size + num_oov_buckets - 1]`.

  The underlying table must be initialized by calling
  `session.run(tf.compat.v1.tables_initializer())` or
  `session.run(table.init())` once.

  Elements in `vocabulary_list` cannot have duplicates, otherwise when executing
  the table initializer op, it will throw a `FailedPreconditionError`.

  Sample Usages:

  ```python
  vocabulary_list = tf.constant(["emerson", "lake", "palmer"])
  table = tf.lookup.index_table_from_tensor(
      vocabulary_list=vocabulary_list, num_oov_buckets=1, default_value=-1)
  features = tf.constant(["emerson", "lake", "and", "palmer"])
  ids = table.lookup(features)
  ...
  tf.compat.v1.tables_initializer().run()

  ids.eval()  ==> [0, 1, 4, 2]
  ```

  Args:
    vocabulary_list: A 1-D `Tensor` that specifies the mapping of keys to
      indices. The type of this object must be castable to `dtype`.
    num_oov_buckets: The number of out-of-vocabulary buckets.
    default_value: The value to use for out-of-vocabulary feature values.
      Defaults to -1.
    hasher_spec: A `HasherSpec` to specify the hash function to use for
      assignment of out-of-vocabulary buckets.
    dtype: The type of values passed to `lookup`. Only string and integers are
      supported.
    name: A name for this op (optional).

  Returns:
    The lookup table to map an input `Tensor` to index `int64` `Tensor`.

  Raises:
    ValueError: If `vocabulary_list` is invalid.
    ValueError: If `num_oov_buckets` is negative.
  """
    if vocabulary_list is None:
        raise ValueError('`vocabulary_list` must be specified.')
    if num_oov_buckets < 0:
        raise ValueError('`num_oov_buckets` must be greater or equal than 0, got %d.' % num_oov_buckets)
    if not dtype.is_integer and dtypes.string != dtype.base_dtype:
        raise TypeError('`dtype` must either be integer or string.')
    with ops.name_scope(name, 'string_to_index'):
        keys = ops.convert_to_tensor(vocabulary_list)
        if keys.dtype.is_integer != dtype.is_integer:
            raise ValueError('Invalid `dtype`: Expected %s, got %s.' % ('integer' if dtype.is_integer else 'non-integer', keys.dtype))
        if not dtype.is_integer and keys.dtype.base_dtype != dtype:
            raise ValueError('Invalid `dtype`: Expected %s, got %s.' % (dtype, keys.dtype))
        num_elements = array_ops.size(keys)
        values = math_ops.cast(math_ops.range(num_elements), dtypes.int64)
        with ops.name_scope(None, 'hash_table'):
            table_keys = math_ops.cast(keys, dtypes.int64) if keys.dtype.is_integer else keys
            init = KeyValueTensorInitializer(table_keys, values, table_keys.dtype.base_dtype, dtypes.int64, name='table_init')
            table = StaticHashTableV1(init, default_value)
        if num_oov_buckets:
            table = IdTableWithHashBuckets(table, num_oov_buckets=num_oov_buckets, hasher_spec=hasher_spec, key_dtype=dtype)
        return table