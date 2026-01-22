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
@property
def parent_nodes(self):
    """Returns all the `Node`s whose output this node immediately depends on."""
    node_deps = []
    for kt in self.keras_inputs:
        layer = kt._keras_history.layer
        node_index = kt._keras_history.node_index
        if layer is not None:
            node_deps.append(layer._inbound_nodes[node_index])
    return node_deps