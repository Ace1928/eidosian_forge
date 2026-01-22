import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
def random_translated_inputs(inputs):
    """Translated inputs with random ops."""
    original_shape = inputs.shape
    unbatched = inputs.shape.rank == 3
    if unbatched:
        inputs = tf.expand_dims(inputs, 0)
    inputs_shape = tf.shape(inputs)
    batch_size = inputs_shape[0]
    img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
    img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
    height_translate = self._random_generator.random_uniform(shape=[batch_size, 1], minval=self.height_lower, maxval=self.height_upper, dtype=tf.float32)
    height_translate = height_translate * img_hd
    width_translate = self._random_generator.random_uniform(shape=[batch_size, 1], minval=self.width_lower, maxval=self.width_upper, dtype=tf.float32)
    width_translate = width_translate * img_wd
    translations = tf.cast(tf.concat([width_translate, height_translate], axis=1), dtype=tf.float32)
    output = transform(inputs, get_translation_matrix(translations), interpolation=self.interpolation, fill_mode=self.fill_mode, fill_value=self.fill_value)
    if unbatched:
        output = tf.squeeze(output, 0)
    output.set_shape(original_shape)
    return output