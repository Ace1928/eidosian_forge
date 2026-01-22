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
def get_current_replica_id_as_int():
    """Returns the current replica ID as an integer, or `None`."""
    replica_context = distribute_lib.get_replica_context()
    if replica_context:
        replica_id = replica_context._replica_id
        if not isinstance(replica_id, int):
            replica_id = tensor_util.constant_value(replica_id)
    else:
        replica_id = distribute_lib.get_update_replica_id()
    return replica_id