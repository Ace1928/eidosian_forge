import functools
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.layers.rnn import rnn_utils
from keras.src.saving import serialization_lib
from keras.src.utils import generic_utils
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def get_batch_input_shape(batch_size, dim):
    shape = tf.TensorShape(dim).as_list()
    return tuple([batch_size] + shape)