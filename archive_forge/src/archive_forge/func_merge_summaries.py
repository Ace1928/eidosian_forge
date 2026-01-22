import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import layer_utils
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def merge_summaries(prev_summary, next_summary, epsilon):
    """Weighted merge sort of summaries.

    Given two summaries of distinct data, this function merges (and compresses)
    them to stay within `epsilon` error tolerance.

    Args:
        prev_summary: 2D `np.ndarray` summary to be merged with `next_summary`.
        next_summary: 2D `np.ndarray` summary to be merged with `prev_summary`.
        epsilon: A float that determines the approxmiate desired precision.

    Returns:
        A 2-D `np.ndarray` that is a merged summary. First column is the
        interpolated partition values, the second is the weights (counts).
    """
    merged = tf.concat((prev_summary, next_summary), axis=1)
    merged = tf.gather(merged, tf.argsort(merged[0]), axis=1)
    return compress(merged, epsilon)