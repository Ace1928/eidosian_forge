from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import concrete_function
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.eager.polymorphic_function import transform
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import gradients_util
from tensorflow.python.util import keras_deps
from tensorflow.python.util import tf_contextlib
@tf_contextlib.contextmanager
def clear_control_inputs():
    """Clears the control inputs but preserves the ControlFlowContext.

  This is needed to preserve the XLAControlFlowControl when clearing
  control inputs for the gradient accumulators in while_v2.
  `ops.control_dependencies` does not allow that.

  Yields:
    A context manager in which the ops created will not have any control inputs
    by default but the control flow context is the same.
  """
    control_flow_context = ops.get_default_graph()._get_control_flow_context()
    with ops.control_dependencies(None):
        ops.get_default_graph()._set_control_flow_context(control_flow_context)
        yield