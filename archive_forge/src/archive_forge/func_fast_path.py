import functools
import itertools
import operator
import tensorflow as tf
from keras.src.backend.tensorflow.core import convert_to_tensor
def fast_path():
    return tf.transpose(tf.gather_nd(input_arr, indices))