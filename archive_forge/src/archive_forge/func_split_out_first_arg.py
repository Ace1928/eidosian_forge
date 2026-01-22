import copy
import functools
import re
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
def split_out_first_arg(self, args, kwargs):
    """Splits (args, kwargs) into (inputs, args, kwargs)."""
    if args:
        inputs = args[0]
        args = args[1:]
    elif self._arg_names[0] in kwargs:
        kwargs = copy.copy(kwargs)
        inputs = kwargs.pop(self._arg_names[0])
    else:
        raise ValueError('The first argument to `Layer.call` must always be passed.')
    return (inputs, args, kwargs)