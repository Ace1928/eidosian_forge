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
def cond_v2(pred, true_fn, false_fn, name='cond'):
    """Like tf.cond, except emits a single If op."""
    if isinstance(pred, bool):
        raise TypeError('pred must not be a Python bool', pred)
    if not name:
        name = 'cond'
    with ops.name_scope(name) as scope:
        true_name = util.unique_fn_name(scope, 'true')
        false_name = util.unique_fn_name(scope, 'false')
        add_control_dependencies = ops.get_default_graph()._add_control_dependencies
        pred = ops.convert_to_tensor(pred)
        if tensor_util.is_tf_type(pred) and (pred.shape.dims is None or pred.shape.dims):
            pred = array_ops.squeeze_v2(pred)
        true_graph = func_graph_module.func_graph_from_py_func(true_name, true_fn, [], {}, func_graph=util.CondBranchFuncGraph(true_name, collections=ops.get_default_graph()._collections), add_control_dependencies=add_control_dependencies, op_return_value=pred)
        false_graph = func_graph_module.func_graph_from_py_func(false_name, false_fn, [], {}, func_graph=util.CondBranchFuncGraph(false_name, collections=ops.get_default_graph()._collections), add_control_dependencies=add_control_dependencies, op_return_value=pred)
        verify_captures(_COND, [true_graph, false_graph])
        return _build_cond(pred, true_graph, false_graph, true_graph.external_captures, false_graph.external_captures, building_gradient=False, name=scope)