import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.types import core
def create_autocast_variable(variable):
    """Creates an AutoCastVariable that wraps another variable.

  This typically just returns `AutoCastVariable(variable)`. But, if the variable
  is a DistributedVariable or one of its subclasses, we instead dynamically
  create a class that subclasses from both AutoCastVariable and
  variable.__class__. This is so the returned variable will still pass
  `isinstance(variable, variable.__class__)`, which is required for
  DistributedVariables and its subclasses to work properly.

  Args:
    variable: A floating-point resource variable to wrap.

  Returns:
    An AutoCastVariable that wraps the variable.
  """
    if not distributed_training_utils.is_distributed_variable(variable):
        return AutoCastVariable(variable)

    class AutoCastDistributedVariable(AutoCastVariable, variable.__class__):
        """An AutoCastVariable that also subclasses from variable.__class__.

    variable.__class__ is either a DistributedVariable or an
    AggregatingVariable.
    """

        def __repr__(self):
            return '<AutoCastDistributedVariable dtype={v.dtype.name} dtype_to_cast_to={v._cast_dtype.name} inner_variable={v._variable}>'.format(v=self)
    return AutoCastDistributedVariable(variable)