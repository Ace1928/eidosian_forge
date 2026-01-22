import contextlib
import warnings
import numpy as np
import tensorflow as tf
import tree
from packaging.version import Version
from tensorflow.python.eager import context as tf_context
from keras.src import callbacks as callbacks_module
from keras.src import metrics as metrics_module
from keras.src import optimizers as optimizers_module
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
def potentially_ragged_concat(tensors):
    """Concats `Tensor`s along their first dimension.

    Args:
        tensors: List of `Tensor`s.

    Returns:
        Concatenation of the inputs along the first dimension -- of type
        `np.ndarray` if all input shapes are compatible, or `tf.RaggedTensor`
        if not.
    """
    if len(tensors) == 1:
        return tensors[0]
    elif isinstance(tensors[0], tf.SparseTensor):
        return tf.sparse.concat(axis=0, sp_inputs=tensors)
    elif isinstance(tensors[0], tf.RaggedTensor):
        return tf.concat(tensors, axis=0)
    non_batch_shapes = tf.stack([tf.shape(tensor)[1:] for tensor in tensors])
    constant_dims = tf.math.reduce_all(non_batch_shapes == non_batch_shapes[:1], axis=0)
    if tf.math.reduce_all(constant_dims).numpy().item():
        if _is_scalar(tensors[0]):
            return tf.stack(tensors, axis=0)
        else:
            return tf.concat(tensors, axis=0)
    constant_inner_dimensions = constant_dims.numpy().tolist()[::-1].index(False)
    if constant_inner_dimensions == 0:
        constant_inner_shape = None
    else:
        constant_inner_shape = tensors[0].shape[-constant_inner_dimensions:]
    return tf.ragged.constant([tensor.numpy() for tensor in tensors], inner_shape=constant_inner_shape).merge_dims(0, 1)