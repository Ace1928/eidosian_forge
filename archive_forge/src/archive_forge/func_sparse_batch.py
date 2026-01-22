import abc
import functools
import queue
import threading
import warnings
import numpy as np
from tensorflow.core.framework import dataset_metadata_pb2
from tensorflow.core.framework import dataset_options_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_autograph
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.data.util import traverse
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd_utils
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as core_random_seed
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base as tracking_base
from tensorflow.python.trackable import resource as resource_lib
from tensorflow.python.types import data as data_types
from tensorflow.python.types import trace
from tensorflow.python.util import deprecation
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest as tf_nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def sparse_batch(self, batch_size, row_shape, name=None):
    """Combines consecutive elements into `tf.sparse.SparseTensor`s.

    Like `Dataset.padded_batch()`, this transformation combines multiple
    consecutive elements of the dataset, which might have different
    shapes, into a single element. The resulting element has three
    components (`indices`, `values`, and `dense_shape`), which
    comprise a `tf.sparse.SparseTensor` that represents the same data. The
    `row_shape` represents the dense shape of each row in the
    resulting `tf.sparse.SparseTensor`, to which the effective batch size is
    prepended. For example:

    ```python
    # NOTE: The following examples use `{ ... }` to represent the
    # contents of a dataset.
    a = { ['a', 'b', 'c'], ['a', 'b'], ['a', 'b', 'c', 'd'] }

    a.apply(tf.data.experimental.dense_to_sparse_batch(
        batch_size=2, row_shape=[6])) ==
    {
        ([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],  # indices
         ['a', 'b', 'c', 'a', 'b'],                 # values
         [2, 6]),                                   # dense_shape
        ([[0, 0], [0, 1], [0, 2], [0, 3]],
         ['a', 'b', 'c', 'd'],
         [1, 6])
    }
    ```

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      row_shape: A `tf.TensorShape` or `tf.int64` vector tensor-like object
        representing the equivalent dense shape of a row in the resulting
        `tf.sparse.SparseTensor`. Each element of this dataset must have the
        same rank as `row_shape`, and must have size less than or equal to
        `row_shape` in each dimension.
      name: (Optional.) A string indicating a name for the `tf.data` operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    from tensorflow.python.data.ops import sparse_batch_op
    return sparse_batch_op._sparse_batch(self, batch_size, row_shape, name)