import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.dtensor import utils
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import control_flow_util
from keras.src.utils import tf_utils
from tensorflow.python.ops.control_flow_ops import (
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import keras_export
def calculate_update_delta():
    decay = tf.convert_to_tensor(1.0 - momentum, name='decay')
    if decay.dtype != variable.dtype.base_dtype:
        decay = tf.cast(decay, variable.dtype.base_dtype)
    update_delta = (variable - tf.cast(value, variable.dtype)) * decay
    if inputs_size is not None:
        update_delta = tf.where(inputs_size > 0, update_delta, backend.zeros_like(update_delta))
    return update_delta