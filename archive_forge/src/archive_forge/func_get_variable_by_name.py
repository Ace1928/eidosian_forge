from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def get_variable_by_name(var_name):
    """Given a variable name, retrieves a handle on the tensorflow Variable."""
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

    def _filter_fn(item):
        try:
            return var_name == item.op.name
        except AttributeError:
            return False
    candidate_vars = list(filter(_filter_fn, global_vars))
    if len(candidate_vars) >= 1:
        candidate_vars = [v for v in candidate_vars if v.trainable]
    else:
        raise ValueError('Unsuccessful at finding variable {}.'.format(var_name))
    if len(candidate_vars) == 1:
        return candidate_vars[0]
    elif len(candidate_vars) > 1:
        raise ValueError('Unsuccessful at finding trainable variable {}. Number of candidates: {}. Candidates: {}'.format(var_name, len(candidate_vars), candidate_vars))
    else:
        return None