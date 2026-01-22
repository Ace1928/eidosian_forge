import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
def get_translation_matrix(translations, name=None):
    """Returns projective transform(s) for the given translation(s).

    Args:
        translations: A matrix of 2-element lists representing `[dx, dy]`
            to translate for each image (for a batch of images).
        name: The name of the op.

    Returns:
        A tensor of shape `(num_images, 8)` projective transforms
            which can be given to `transform`.
    """
    with backend.name_scope(name or 'translation_matrix'):
        num_translations = tf.shape(translations)[0]
        return tf.concat(values=[tf.ones((num_translations, 1), tf.float32), tf.zeros((num_translations, 1), tf.float32), -translations[:, 0, None], tf.zeros((num_translations, 1), tf.float32), tf.ones((num_translations, 1), tf.float32), -translations[:, 1, None], tf.zeros((num_translations, 2), tf.float32)], axis=1)