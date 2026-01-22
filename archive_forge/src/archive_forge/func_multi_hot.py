import warnings
import tensorflow as tf
from keras.src.backend import standardize_data_format
from keras.src.backend import standardize_dtype
from keras.src.backend.common.backend_utils import (
from keras.src.backend.config import epsilon
from keras.src.backend.tensorflow.core import cast
from keras.src.backend.tensorflow.core import convert_to_tensor
def multi_hot(x, num_classes, axis=-1, dtype='float32'):
    x = convert_to_tensor(x)
    reduction_axis = 1 if len(x.shape) > 1 else 0
    outputs = tf.reduce_max(one_hot(cast(x, 'int32'), num_classes, axis=axis, dtype=dtype), axis=reduction_axis)
    return outputs