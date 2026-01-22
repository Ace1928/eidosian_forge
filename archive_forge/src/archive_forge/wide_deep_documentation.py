import tensorflow.compat.v2 as tf
from keras.src import activations
from keras.src import backend
from keras.src import layers as layer_module
from keras.src.engine import base_layer
from keras.src.engine import data_adapter
from keras.src.engine import training as keras_training
from keras.src.saving import serialization_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import keras_export
Create a Wide & Deep Model.

        Args:
          linear_model: a premade LinearModel, its output must match the output
            of the dnn model.
          dnn_model: a `tf.keras.Model`, its output must match the output of the
            linear model.
          activation: Activation function. Set it to None to maintain a linear
            activation.
          **kwargs: The keyword arguments that are passed on to
            BaseLayer.__init__. Allowed keyword arguments include `name`.
        