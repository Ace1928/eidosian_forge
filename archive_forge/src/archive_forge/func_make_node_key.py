import collections
import tree
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.config import backend
from keras.src.ops.operation import Operation
from keras.src.utils.nest import pack_sequence_as
def make_node_key(op, node_index):
    return str(id(op)) + '_ib-' + str(node_index)