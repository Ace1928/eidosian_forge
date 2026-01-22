import collections
import tensorflow.compat.v2 as tf
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.layers.rnn import gru_lstm_utils
from keras.src.layers.rnn.base_cudnn_rnn import _CuDNNRNN
from tensorflow.python.util.tf_export import keras_export
@property
def cell(self):
    return self._cell