from tensorflow.python.data.experimental.ops.cardinality import assert_cardinality
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
Creates a table initializer from a `tf.data.Dataset`.

    Args:
      dataset: A `tf.data.Dataset` object that produces tuples of scalars. The
        first scalar is treated as a key and the second as value.
    Raises: ValueError if `dataset` doesn't conform to specifications.
    Returns: A `DatasetInitializer` object
    