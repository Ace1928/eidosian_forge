from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.training.saving import saveable_object
def get_on_read_saveable(var, primary_var, name):
    """Return saveables for ON_READ variable."""

    def tensor():
        return var._get_cross_replica()
    spec = saveable_object.SaveSpec(tensor=tensor, slice_spec='', name=name, dtype=var.dtype, device=primary_var.device)
    return (tensor, [spec])