import collections
import copy
import json
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest
def iterate_inbound(self):
    """Yields tuples representing the data inbound from other nodes.

    Yields:
      tuples like: (inbound_layer, node_index, tensor_index, tensor).
    """
    for kt in self.keras_inputs:
        keras_history = kt._keras_history
        layer = keras_history.layer
        node_index = keras_history.node_index
        tensor_index = keras_history.tensor_index
        yield (layer, node_index, tensor_index, kt)