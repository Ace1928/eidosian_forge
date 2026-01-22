from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
@property
def grad_sync(self):
    """A control trigger node for synchronization in the grad loop.

    One main use is to keep the pop ops of a stack executed in the
    iteration order.
    """
    if self._grad_sync is None:
        with ops.control_dependencies(None):
            self._grad_sync = control_flow_ops.control_trigger(name='b_sync')
        self._grad_sync._set_control_flow_context(self._grad_context)
        self._grad_index.op._add_control_input(self._grad_sync)
        if self._grad_context.outer_context:
            self._grad_context.outer_context.AddInnerOp(self._grad_sync)
    return self._grad_sync