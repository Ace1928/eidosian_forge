import copy
from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
def strip_unused_from_files(input_graph, input_binary, output_graph, output_binary, input_node_names, output_node_names, placeholder_type_enum):
    """Removes unused nodes from a graph file."""
    if not gfile.Exists(input_graph):
        print("Input graph file '" + input_graph + "' does not exist!")
        return -1
    if not output_node_names:
        print('You need to supply the name of a node to --output_node_names.')
        return -1
    input_graph_def = graph_pb2.GraphDef()
    mode = 'rb' if input_binary else 'r'
    with gfile.GFile(input_graph, mode) as f:
        if input_binary:
            input_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_graph_def)
    output_graph_def = strip_unused(input_graph_def, input_node_names.split(','), output_node_names.split(','), placeholder_type_enum)
    if output_binary:
        with gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
    else:
        with gfile.GFile(output_graph, 'w') as f:
            f.write(text_format.MessageToString(output_graph_def))
    print('%d ops in the final graph.' % len(output_graph_def.node))