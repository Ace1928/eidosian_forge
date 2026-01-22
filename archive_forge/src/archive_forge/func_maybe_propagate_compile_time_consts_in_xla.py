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
def maybe_propagate_compile_time_consts_in_xla(op):
    """Tells XLA whether to propagate compile-time consts in the loop body.

  This is needed to make compile time constants available to ops, for example
  `max_num_elements` in `EmptyTensorList`, inside the loop body. Ideally this
  would always be turned on, but that doesn't work with legacy functionalized
  while_loops.

  Args:
    op: A `While` Operation.
  """
    if control_flow_util.GraphOrParentsInXlaContext(op.graph):
        op._set_attr('_xla_propagate_compile_time_consts', attr_value_pb2.AttrValue(b=True))