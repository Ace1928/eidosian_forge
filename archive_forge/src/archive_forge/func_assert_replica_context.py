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
def assert_replica_context(strategy):
    replica_context = distribute_lib.get_replica_context()
    if not replica_context:
        raise RuntimeError('Replica-local variables may only be assigned in a replica context.')
    if replica_context.strategy is not strategy:
        raise RuntimeError('Replica-local variables may only be assigned in a replica context.')