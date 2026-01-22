import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import tf_utils
def compute_shape_for_encode_categorical(shape, output_mode, depth):
    """Computes the output shape of `encode_categorical_inputs`."""
    if output_mode == INT:
        return tf.TensorShape(shape)
    if not shape:
        return tf.TensorShape([depth])
    if output_mode == ONE_HOT and shape[-1] != 1:
        return tf.TensorShape(shape + [depth])
    else:
        return tf.TensorShape(shape[:-1] + [depth])