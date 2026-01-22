import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
def random_height_inputs(inputs):
    """Inputs height-adjusted with random ops."""
    inputs_shape = tf.shape(inputs)
    img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
    img_wd = inputs_shape[W_AXIS]
    height_factor = self._random_generator.random_uniform(shape=[], minval=1.0 + self.height_lower, maxval=1.0 + self.height_upper)
    adjusted_height = tf.cast(height_factor * img_hd, tf.int32)
    adjusted_size = tf.stack([adjusted_height, img_wd])
    output = tf.image.resize(images=inputs, size=adjusted_size, method=self._interpolation_method)
    output = tf.cast(output, self.compute_dtype)
    output_shape = inputs.shape.as_list()
    output_shape[H_AXIS] = None
    output.set_shape(output_shape)
    return output