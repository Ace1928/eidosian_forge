import functools
import sys
import time
import six
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.distribute_lib import InputReplicationMode
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import distribute as distribute_types
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
def create_dummy_tensor(spec):
    """Create a dummy tensor with possible batch dimensions set to 0."""
    if hasattr(spec, '_create_empty_value'):
        return spec._create_empty_value()
    if isinstance(spec, ragged_tensor.RaggedTensorSpec):
        feature_shape = spec._shape[:1].concatenate(spec._shape[1 + spec._ragged_rank:])
        feature_type = spec._dtype
    else:
        feature_shape = spec.shape
        feature_type = spec.dtype
    dims = [dim if dim is not None else 0 for dim in feature_shape.as_list()] if feature_shape else []
    if dims and (isinstance(spec, ragged_tensor.RaggedTensorSpec) or feature_shape.is_fully_defined()):
        dims[0] = tensor_shape.Dimension(0)
    if isinstance(spec, sparse_tensor.SparseTensorSpec):
        return sparse_tensor.SparseTensor(values=array_ops.zeros(0, feature_type), indices=array_ops.zeros((0, len(dims)), dtypes.int64), dense_shape=dims)
    dummy_tensor = array_ops.zeros(tensor_shape.TensorShape(dims), feature_type)
    if isinstance(spec, ragged_tensor.RaggedTensorSpec):
        row_splits = array_ops.zeros(1, spec._row_splits_dtype)
        dummy_tensor = ragged_tensor.RaggedTensor.from_nested_row_splits(dummy_tensor, (row_splits,) * spec._ragged_rank, validate=False)
    return dummy_tensor