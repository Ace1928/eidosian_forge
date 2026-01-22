from typing import TYPE_CHECKING, Dict, List, Optional, Union, Tuple
import numpy as np
import pyarrow
import tensorflow as tf
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
def convert_ndarray_to_tf_tensor(ndarray: np.ndarray, dtype: Optional[tf.dtypes.DType]=None, type_spec: Optional[tf.TypeSpec]=None) -> tf.Tensor:
    """Convert a NumPy ndarray to a TensorFlow Tensor.

    Args:
        ndarray: A NumPy ndarray that we wish to convert to a TensorFlow Tensor.
        dtype: A TensorFlow dtype for the created tensor; if None, the dtype will be
            inferred from the NumPy ndarray data.
        type_spec: A type spec that specifies the shape and dtype of the returned
            tensor. If you specify ``dtype``, the dtype stored in the type spec is
            ignored.

    Returns: A TensorFlow Tensor.
    """
    if dtype is None and type_spec is not None:
        dtype = type_spec.dtype
    is_ragged = isinstance(type_spec, tf.RaggedTensorSpec)
    ndarray = _unwrap_ndarray_object_type_if_needed(ndarray)
    if is_ragged:
        return tf.ragged.constant(ndarray, dtype=dtype)
    else:
        return tf.convert_to_tensor(ndarray, dtype=dtype)