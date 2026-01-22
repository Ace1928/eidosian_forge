import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import dataset_utils
from tensorflow.python.util.tf_export import keras_export
def path_to_string_content(path, max_length):
    txt = tf.io.read_file(path)
    if max_length is not None:
        txt = tf.compat.v1.strings.substr(txt, 0, max_length)
    return txt