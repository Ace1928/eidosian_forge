from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.keras import backend
from tensorflow.python.ops import variables
def is_distributed_variable(v):
    """Returns whether `v` is a distributed variable."""
    return isinstance(v, values_lib.DistributedValues) and isinstance(v, variables.Variable)