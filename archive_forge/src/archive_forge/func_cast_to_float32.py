import numpy as np
import tensorflow as tf
from tensorflow import nest
def cast_to_float32(tensor):
    if tensor.dtype == tf.float32:
        return tensor
    if tensor.dtype == tf.string:
        return tf.strings.to_number(tensor, tf.float32)
    return tf.cast(tensor, tf.float32)