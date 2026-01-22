import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.util import nest
def is_adjoint_pair(x, y):
    """True iff x and y are adjoints of each other (by id, not entries)."""
    if x is y:
        if x.is_self_adjoint is False:
            return False
        if x.is_self_adjoint:
            return True
    return x.H is y or y.H is x