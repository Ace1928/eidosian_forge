import copy
import functools
import re
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def readable_memory_size(weight_memory_size):
    """Convert the weight memory size (Bytes) to a readable string."""
    units = ['Byte', 'KB', 'MB', 'GB', 'TB', 'PB']
    scale = 1024
    for unit in units:
        if weight_memory_size / scale < 1:
            return '{:.2f} {}'.format(weight_memory_size, unit)
        else:
            weight_memory_size /= scale
    return '{:.2f} {}'.format(weight_memory_size, units[-1])