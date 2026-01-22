import math
import warnings
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.dtensor import utils
from keras.src.saving import serialization_lib
from tensorflow.python.util.tf_export import keras_export
Returns a tensor object initialized to a 2D identity matrix.

        Args:
          shape: Shape of the tensor. It should have exactly rank 2.
          dtype: Optional dtype of the tensor. Only floating point types are
           supported. If not specified, `tf.keras.backend.floatx()` is used,
           which default to `float32` unless you configured it otherwise
           (via `tf.keras.backend.set_floatx(float_dtype)`)
          **kwargs: Additional keyword arguments.
        