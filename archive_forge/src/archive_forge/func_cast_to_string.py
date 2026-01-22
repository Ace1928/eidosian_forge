import numpy as np
import tensorflow as tf
from tensorflow import nest
def cast_to_string(tensor):
    if tensor.dtype == tf.string:
        return tensor
    return tf.strings.as_string(tensor)