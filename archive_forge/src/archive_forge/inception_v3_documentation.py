import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
Utility function to apply conv + BN.

    Args:
      x: input tensor.
      filters: filters in `Conv2D`.
      num_row: height of the convolution kernel.
      num_col: width of the convolution kernel.
      padding: padding mode in `Conv2D`.
      strides: strides in `Conv2D`.
      name: name of the ops; will become `name + '_conv'`
        for the convolution and `name + '_bn'` for the
        batch norm layer.

    Returns:
      Output tensor after applying `Conv2D` and `BatchNormalization`.
    