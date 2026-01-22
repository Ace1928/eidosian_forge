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
def run_as_function_for_tape_gradients(make_op, inputs):
    """Fix higher-order tape gradients by wrapping `make_op` in a function.

  Args:
    make_op: A function that takes a list of inputs and returns a list of output
      tensors. This function should set any handle data relevant to its outputs
      before returning.
    inputs: A list of tensors to check for tape gradients and pass to
      `make_op`. These should include all tensors used in `make_op`.

  Returns:
    Tensors corresponding to `make_op`'s output.
  """
    if gradients_util.PossibleTapeGradientTypes(inputs) == gradients_util.POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER and (not (ops.get_default_graph().building_function and 'cflow_gradient_wrapper' in ops.get_default_graph().name)):
        results = tracing_compilation.call_function((inputs,), tracing_options=tracing_compilation.TracingOptions(make_op, 'cflow_gradient_wrapper', autograph=False))
        return results
    else:
        return make_op(inputs)