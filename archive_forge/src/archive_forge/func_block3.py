import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def block3(x, filters, kernel_size=3, stride=1, groups=32, conv_shortcut=True, name=None):
    """A residual block.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      groups: default 32, group size for grouped convolution.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    if conv_shortcut:
        shortcut = layers.Conv2D(64 // groups * filters, 1, strides=stride, use_bias=False, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-05, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x
    x = layers.Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-05, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)
    c = filters // groups
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=c, use_bias=False, name=name + '_2_conv')(x)
    x_shape = backend.shape(x)[:-1]
    x = backend.reshape(x, backend.concatenate([x_shape, (groups, c, c)]))
    x = layers.Lambda(lambda x: sum((x[:, :, :, :, i] for i in range(c))), name=name + '_2_reduce')(x)
    x = backend.reshape(x, backend.concatenate([x_shape, (filters,)]))
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-05, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)
    x = layers.Conv2D(64 // groups * filters, 1, use_bias=False, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-05, name=name + '_3_bn')(x)
    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x