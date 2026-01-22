import os
import sys
from keras.src.api_export import keras_export
from keras.src.utils import io_utils
def make_node(layer, **kwargs):
    node = pydot.Node(str(id(layer)), label=make_layer_label(layer, **kwargs))
    node.set('fontname', 'Helvetica')
    node.set('border', '0')
    node.set('margin', '0')
    return node