import itertools
import threading
import types
from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.util import tf_decorator
def layer_uses_training_bool(layer):
    """Returns whether this layer or any of its children uses the training arg."""
    if layer._expects_training_arg:
        return True
    visited = {layer}
    to_visit = list_all_layers(layer)
    while to_visit:
        layer = to_visit.pop()
        if layer in visited:
            continue
        if getattr(layer, '_expects_training_arg', True):
            return True
        visited.add(layer)
        to_visit.extend(list_all_layers(layer))
    return False