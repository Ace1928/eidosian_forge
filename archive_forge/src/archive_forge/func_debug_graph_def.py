from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
@property
def debug_graph_def(self):
    """The debugger-decorated GraphDef."""
    return self._debug_graph_def