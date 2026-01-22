import collections
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.rnn import rnn_utils
from keras.src.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from keras.src.layers.rnn.stacked_rnn_cells import StackedRNNCells
from keras.src.saving import serialization_lib
from keras.src.saving.legacy.saved_model import layer_serialization
from keras.src.utils import generic_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
def get_state_spec(shape):
    state_spec_shape = tf.TensorShape(shape).as_list()
    state_spec_shape = [None] + state_spec_shape
    return InputSpec(shape=tuple(state_spec_shape))