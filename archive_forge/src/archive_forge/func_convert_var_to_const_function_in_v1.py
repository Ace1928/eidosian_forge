import collections
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def convert_var_to_const_function_in_v1(func, lower_control_flow=True, aggressive_inlining=False):
    """Replaces all the variables in a graph with constants of the same values.

  This function works as same as convert_variables_to_constants_v2, but it
  should be used in Graph mode. It is a temporary solution when users want to
  integrate their models written in TF2 with infra that requires TF1 mode.

  The current implementation only works for graphs that do not contain any
  control flow or embedding related ops.

  The function must be called in a Session context.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Boolean indicating whether or not to do aggressive
      function inlining (might be unsafe if function has stateful ops, not
      properly connected to control outputs). (default False)

  Raises:
      RuntimeError: If no Session context is present.

  Returns:
    ConcreteFunction containing a simplified version of the original.
  """
    session = ops.get_default_session()
    if session is None:
        raise RuntimeError('The conversion must be carried out in a Session context.')
    converter_data = _FunctionConverterDataInGraph(func=func, lower_control_flow=lower_control_flow, aggressive_inlining=aggressive_inlining, session=session)
    output_graph_def, converted_input_indices = _replace_variables_by_constants(converter_data=converter_data)
    return _construct_concrete_function(func, output_graph_def, converted_input_indices)