import abc
import math
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest
def compute_step(completed_fraction, geometric=False):
    """Helper for `cond` operation."""
    if geometric:
        i_restart = math_ops.floor(math_ops.log(1.0 - completed_fraction * (1.0 - t_mul)) / math_ops.log(t_mul))
        sum_r = (1.0 - t_mul ** i_restart) / (1.0 - t_mul)
        completed_fraction = (completed_fraction - sum_r) / t_mul ** i_restart
    else:
        i_restart = math_ops.floor(completed_fraction)
        completed_fraction -= i_restart
    return (i_restart, completed_fraction)