import functools
import weakref
import numpy as np
from tensorflow.python.util import nest
def print_layer_summary_with_connections(layer):
    """Prints a summary for a single layer (including topological connections).

    Args:
        layer: target layer.
    """
    try:
        output_shape = layer.output_shape
    except AttributeError:
        output_shape = 'multiple'
    connections = []
    for node in layer._inbound_nodes:
        if relevant_nodes and node not in relevant_nodes:
            continue
        for inbound_layer, node_index, tensor_index, _ in node.iterate_inbound():
            connections.append('{}[{}][{}]'.format(inbound_layer.name, node_index, tensor_index))
    name = layer.name
    cls_name = layer.__class__.__name__
    if not connections:
        first_connection = ''
    else:
        first_connection = connections[0]
    fields = [name + ' (' + cls_name + ')', output_shape, layer.count_params(), first_connection]
    print_row(fields, positions)
    if len(connections) > 1:
        for i in range(1, len(connections)):
            fields = ['', '', '', connections[i]]
            print_row(fields, positions)