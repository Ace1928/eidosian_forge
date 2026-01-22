import uuid
import tensorflow.compat.v2 as tf
from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine import base_layer
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.rnn import gru_lstm_utils
from keras.src.layers.rnn import rnn_utils
from keras.src.layers.rnn.base_rnn import RNN
from keras.src.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def stardard_lstm_fn():
    return standard_lstm(inputs=inputs, init_h=init_h, init_c=init_c, kernel=kernel, recurrent_kernel=recurrent_kernel, bias=bias, mask=mask, time_major=time_major, go_backwards=go_backwards, sequence_lengths=sequence_lengths, zero_output_for_mask=zero_output_for_mask, return_sequences=return_sequences)