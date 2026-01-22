import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
def get_batch_dim(type_spec):
    try:
        shape = type_spec._to_legacy_output_shapes()
    except NotImplementedError:
        return None
    if not isinstance(shape, tensor_shape.TensorShape):
        return None
    if shape.rank is None:
        return None
    if len(shape) < 1:
        raise ValueError('Invalid `batch_sizes`. Expected dataset with rank of >= 1 but found a dataset with scalar elements. Fix the issue by adding the `batch` transformation to the dataset.')
    return shape.dims[0].value