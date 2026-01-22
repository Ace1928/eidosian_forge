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
def output_all_intermediates():
    """Whether to output all intermediates of a functional control flow op.

  The default behavior is to output intermediates only when building a Keras
  Layer in graph mode and that too when certain other conditions are met:
  1. We do not output intermediates if the functional control flow op
     is being built inside a FuncGraph which is not a If/While graph. This
     guards against outputting intermediates in eager mode since keras adds
     tensors to a FuncGraph named "keras_graph" in that case. Also because we
     do not output intermediates of tf.function (since this feature is only for
     backwards compatibility) outputting intermediates of functional control
     flow ops built inside tf.function is of no value.
  2. We do not output intermediates when the compilation is using XLA or for a
     TPU.
  3. We do not output intermediates when a single threaded executor is used
     since that does not perform inlining and pruning.

  Returns:
    A bool telling whether to output all intermediates.
  """
    if _EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE is not None:
        return _EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
    if in_defun():
        return False
    if control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
        return False
    if context.context().function_call_options.executor_type == 'SINGLE_THREADED_EXECUTOR':
        return False
    return _is_building_keras_layer()