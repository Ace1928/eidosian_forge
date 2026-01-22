import tensorflow.compat.v2 as tf
from keras.src.engine.base_layer import Layer
from tensorflow.python.util.tf_export import keras_export
Identity layer.

    This layer should be used as a placeholder when no operation is to be
    performed. The layer is argument insensitive, and returns its `inputs`
    argument as output.

    Args:
        name: Optional name for the layer instance.
    