import collections
from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def get_func_graphs(op):
    """Returns `FuncGraph`s for the input op branches.

  Args:
    op: The If or Case Operation.

  Returns:
    A tuple of the `FuncGraph`s of the then_branch and else_branch (all branches
    for Case).
  """

    def _get_func_graph_for_branch(name_attr_list, cached_attr_name=None):
        """Generates and returns a FuncGraph for the given branch."""
        func_graph = None
        if cached_attr_name is not None:
            func_graph = getattr(op, cached_attr_name, None)
        inputs = op.inputs[1:]
        if func_graph is None:
            input_shapes = [t.shape for t in inputs]
            func_graph = util.get_func_graph(op, input_shapes, name_attr_list.name)
        for external_t, internal_t in zip(inputs, func_graph.inputs):
            handle_data_util.copy_handle_data(external_t, internal_t)
        func_graph.function_captures.reset_captures(inputs, func_graph.inputs)
        func_graph._forward_cond = op
        return func_graph
    if op.type in ['If', 'StatelessIf']:
        return (_get_func_graph_for_branch(op.get_attr('then_branch'), '_true_graph'), _get_func_graph_for_branch(op.get_attr('else_branch'), '_false_graph'))
    elif op.type in ['Case', 'StatelessCase']:
        return [_get_func_graph_for_branch(branch_fn, '_branch_graph_{}'.format(i)) for i, branch_fn in enumerate(op.get_attr('branches'))]
    else:
        raise ValueError('Unsupported op type: {}'.format(op.type))