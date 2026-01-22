import tensorflow.compat.v2 as tf
from keras.src import activations
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import conv_utils
def convolution_op(self, inputs, kernel):
    if self.padding == 'causal':
        tf_padding = 'VALID'
    elif isinstance(self.padding, str):
        tf_padding = self.padding.upper()
    else:
        tf_padding = self.padding
    return tf.nn.convolution(inputs, kernel, strides=list(self.strides), padding=tf_padding, dilations=list(self.dilation_rate), data_format=self._tf_data_format, name=self.__class__.__name__)