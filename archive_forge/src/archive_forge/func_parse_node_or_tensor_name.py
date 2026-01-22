from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
def parse_node_or_tensor_name(name):
    """Get the node name from a string that can be node or tensor name.

  Args:
    name: An input node name (e.g., "node_a") or tensor name (e.g.,
      "node_a:0"), as a str.

  Returns:
    1) The node name, as a str. If the input name is a tensor name, i.e.,
      consists of a colon, the final colon and the following output slot
      will be stripped.
    2) If the input name is a tensor name, the output slot, as an int. If
      the input name is not a tensor name, None.
  """
    if ':' in name and (not name.endswith(':')):
        node_name = name[:name.rfind(':')]
        output_slot = int(name[name.rfind(':') + 1:])
        return (node_name, output_slot)
    else:
        return (name, None)