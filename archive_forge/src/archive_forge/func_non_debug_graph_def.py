from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.platform import tf_logging as logging
@property
def non_debug_graph_def(self):
    """The GraphDef without the Copy* and Debug* nodes added by the debugger."""
    self._reconstruct_non_debug_graph_def()
    return self._non_debug_graph_def