import os
import sys
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.util import nest
def format_shape(shape):
    return str(shape).replace(str(None), 'None')