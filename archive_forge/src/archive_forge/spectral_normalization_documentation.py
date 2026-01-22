import tensorflow.compat.v2 as tf
from keras.src.initializers import TruncatedNormal
from keras.src.layers.rnn import Wrapper
from tensorflow.python.util.tf_export import keras_export
Generate spectral normalized weights.

        This method will update the value of `self.kernel` with the
        spectral normalized value, so that the layer is ready for `call()`.
        