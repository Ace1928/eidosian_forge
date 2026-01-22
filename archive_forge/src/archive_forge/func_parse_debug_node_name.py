from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
def parse_debug_node_name(node_name):
    """Parse the name of a debug node.

  Args:
    node_name: Name of the debug node.

  Returns:
    1. Name of the watched node, as a str.
    2. Output slot index of the watched tensor, as an int.
    3. Index of the debug node, as an int.
    4. Name of the debug op, as a str, e.g, "DebugIdentity".

  Raises:
    ValueError: If the input node name is not a valid debug node name.
  """
    prefix = '__dbg_'
    name = node_name
    if not name.startswith(prefix):
        raise ValueError("Invalid prefix in debug node name: '%s'" % node_name)
    name = name[len(prefix):]
    if name.count('_') < 2:
        raise ValueError("Invalid debug node name: '%s'" % node_name)
    debug_op = name[name.rindex('_') + 1:]
    name = name[:name.rindex('_')]
    debug_op_index = int(name[name.rindex('_') + 1:])
    name = name[:name.rindex('_')]
    if name.count(':') != 1:
        raise ValueError("Invalid tensor name in debug node name: '%s'" % node_name)
    watched_node_name = name[:name.index(':')]
    watched_output_slot = int(name[name.index(':') + 1:])
    return (watched_node_name, watched_output_slot, debug_op_index, debug_op)