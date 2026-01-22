import copy
import inspect
import warnings
import tree
from keras.src import backend
from keras.src import ops
from keras.src.backend.common import global_state
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.legacy.saving import saving_utils
from keras.src.legacy.saving import serialization as legacy_serialization
from keras.src.models.model import Model
from keras.src.ops.function import Function
from keras.src.ops.function import make_node_key
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
def map_tensors(tensors):
    if isinstance(tensors, dict):
        return {k: get_tensor_config(v) for k, v in tensors.items()}
    if isinstance(tensors, (list, tuple)):
        return [get_tensor_config(v) for v in tensors]
    else:
        return [get_tensor_config(tensors)]