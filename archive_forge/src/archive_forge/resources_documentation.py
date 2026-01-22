import collections
import os
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_should_use
Initializes the resources in the given list.

  Args:
   resource_list: list of resources to initialize.
   name: name of the initialization op.

  Returns:
   op responsible for initializing all resources.
  