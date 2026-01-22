import warnings
import tensorflow as tf
from keras.src.backend import standardize_data_format
from keras.src.backend import standardize_dtype
from keras.src.backend.common.backend_utils import (
from keras.src.backend.config import epsilon
from keras.src.backend.tensorflow.core import cast
from keras.src.backend.tensorflow.core import convert_to_tensor
def average_pool(inputs, pool_size, strides=None, padding='valid', data_format=None):
    data_format = standardize_data_format(data_format)
    strides = pool_size if strides is None else strides
    padding = padding.upper()
    tf_data_format = _convert_data_format('channels_last', len(inputs.shape))
    if data_format == 'channels_first':
        inputs = _transpose_spatial_inputs(inputs)
    outputs = tf.nn.avg_pool(inputs, pool_size, strides, padding, tf_data_format)
    if data_format == 'channels_first':
        outputs = _transpose_spatial_outputs(outputs)
    return outputs