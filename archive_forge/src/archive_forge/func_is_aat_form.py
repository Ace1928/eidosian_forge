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
def is_aat_form(operators):
    """Returns True if operators is of the form A @ A.H, possibly recursively."""
    operators = list(operators)
    if not operators:
        raise ValueError('AAT form is undefined for empty operators')
    if len(operators) % 2:
        return False
    return all((is_adjoint_pair(operators[i], operators[-1 - i]) for i in range(len(operators) // 2)))