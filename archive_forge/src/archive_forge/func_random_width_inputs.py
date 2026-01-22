import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
def random_width_inputs(inputs):
    """Inputs width-adjusted with random ops."""
    inputs_shape = tf.shape(inputs)
    img_hd = inputs_shape[H_AXIS]
    img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
    width_factor = self._random_generator.random_uniform(shape=[], minval=1.0 + self.width_lower, maxval=1.0 + self.width_upper)
    adjusted_width = tf.cast(width_factor * img_wd, tf.int32)
    adjusted_size = tf.stack([img_hd, adjusted_width])
    output = tf.image.resize(images=inputs, size=adjusted_size, method=self._interpolation_method)
    output = tf.cast(output, self.compute_dtype)
    output_shape = inputs.shape.as_list()
    output_shape[W_AXIS] = None
    output.set_shape(output_shape)
    return output