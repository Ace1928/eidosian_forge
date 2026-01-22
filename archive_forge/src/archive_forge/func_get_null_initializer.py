import collections
import numpy as np
from keras.src import backend
from keras.src.layers.layer import Layer
from keras.src.utils import argument_validation
from keras.src.utils import tf_utils
from keras.src.utils.module_utils import tensorflow as tf
def get_null_initializer(key_dtype, value_dtype):

    class NullInitializer(tf.lookup.KeyValueTensorInitializer):
        """A placeholder initializer for restoring from a SavedModel."""

        def __init__(self, key_dtype, value_dtype):
            """Construct a table initializer object.

            Args:
            key_dtype: Type of the table keys.
            value_dtype: Type of the table values.
            """
            self._key_dtype = key_dtype
            self._value_dtype = value_dtype

        @property
        def key_dtype(self):
            """The expected table key dtype."""
            return self._key_dtype

        @property
        def value_dtype(self):
            """The expected table value dtype."""
            return self._value_dtype

        def initialize(self, table):
            """Returns the table initialization op."""
            pass
    return NullInitializer(key_dtype, value_dtype)