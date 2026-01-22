import copy
import functools
import re
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
@property
def default_training_arg(self):
    """The default value given to the "training" argument."""
    return self._default_training_arg