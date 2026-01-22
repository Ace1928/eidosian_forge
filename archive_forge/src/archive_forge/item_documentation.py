from tensorflow.core.grappler.costs import op_performance_data_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.grappler import _pywrap_tf_item as tf_item
Return a list of hard colocation constraints.

    All the nodes in a colocation tuple must be placed on the same device for
    the model to work.

    Returns:
      A list of colocation tuples.
    