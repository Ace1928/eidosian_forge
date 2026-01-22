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
def convert_revived_tensor(x):
    if isinstance(x, backend.KerasTensor):
        history = x._pre_serialization_keras_history
        if history is None:
            return x
        layer = created_layers.get(history[0], None)
        if layer is None:
            raise ValueError(f'Unknown layer: {history[0]}')
        inbound_node_index = history[1]
        inbound_tensor_index = history[2]
        if len(layer._inbound_nodes) <= inbound_node_index:
            raise ValueError(f'Layer node index out of bounds.\ninbound_layer = {layer}\ninbound_layer._inbound_nodes = {layer._inbound_nodes}\ninbound_node_index = {inbound_node_index}')
        inbound_node = layer._inbound_nodes[inbound_node_index]
        return inbound_node.output_tensors[inbound_tensor_index]
    return x