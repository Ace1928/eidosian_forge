import functools
import weakref
from enum import Enum
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.utils import losses_utils
from keras.src.utils import tf_utils
from keras.src.utils.generic_utils import to_list
def sparse_categorical_matches(y_true, y_pred):
    """Creates float Tensor, 1.0 for label-prediction match, 0.0 for mismatch.

    You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    Args:
      y_true: Integer ground truth values.
      y_pred: The prediction values.

    Returns:
      Match tensor: 1.0 for label-prediction match, 0.0 for mismatch.
    """
    reshape_matches = False
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true)
    y_true_org_shape = tf.shape(y_true)
    y_pred_rank = y_pred.shape.ndims
    y_true_rank = y_true.shape.ndims
    if y_true_rank is not None and y_pred_rank is not None and (len(backend.int_shape(y_true)) == len(backend.int_shape(y_pred))):
        y_true = tf.squeeze(y_true, [-1])
        reshape_matches = True
    y_pred = tf.math.argmax(y_pred, axis=-1)
    if backend.dtype(y_pred) != backend.dtype(y_true):
        y_pred = tf.cast(y_pred, backend.dtype(y_true))
    matches = tf.cast(tf.equal(y_true, y_pred), backend.floatx())
    if reshape_matches:
        matches = tf.reshape(matches, shape=y_true_org_shape)
    return matches