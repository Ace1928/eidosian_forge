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
def serialize_node(node, node_reindexing_map):
    if not node.input_tensors:
        return
    args = node.arguments.args
    kwargs = node.arguments.kwargs
    return {'args': serialization_lib.serialize_keras_object(args), 'kwargs': serialization_lib.serialize_keras_object(kwargs)}