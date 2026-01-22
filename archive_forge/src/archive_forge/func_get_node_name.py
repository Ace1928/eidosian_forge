from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
def get_node_name(element_name):
    node_name, _ = parse_node_or_tensor_name(element_name)
    return node_name