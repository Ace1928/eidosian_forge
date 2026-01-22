from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import hashlib
import numbers
import tensorflow.compat.v2 as tf
from keras.src.layers.rnn.cell_wrappers import _enumerated_map_structure_up_to
from keras.src.layers.rnn.cell_wrappers import _parse_config_to_function
from keras.src.layers.rnn.cell_wrappers import _serialize_function_to_config
from keras.src.layers.rnn.legacy_cells import RNNCell
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.util.tf_export import tf_export
Run the cell on specified device.