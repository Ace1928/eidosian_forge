import functools
import weakref
from enum import Enum
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.utils import losses_utils
from keras.src.utils import tf_utils
from keras.src.utils.generic_utils import to_list
def sparse_top_k_categorical_matches(y_true, y_pred, k=5):
    """Creates float Tensor, 1.0 for label-TopK_prediction match, 0.0 for
    mismatch.

    Args:
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.
      k: (Optional) Number of top elements to look at for computing accuracy.
        Defaults to `5`.

    Returns:
      Match tensor: 1.0 for label-prediction match, 0.0 for mismatch.
    """
    reshape_matches = False
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true_rank = y_true.shape.ndims
    y_pred_rank = y_pred.shape.ndims
    y_true_org_shape = tf.shape(y_true)
    if y_true_rank is not None and y_pred_rank is not None:
        if y_pred_rank > 2:
            y_pred = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
        if y_true_rank > 1:
            reshape_matches = True
            y_true = tf.reshape(y_true, [-1])
    matches = tf.cast(tf.math.in_top_k(predictions=y_pred, targets=tf.cast(y_true, 'int32'), k=k), dtype=backend.floatx())
    if reshape_matches:
        return tf.reshape(matches, shape=y_true_org_shape)
    return matches