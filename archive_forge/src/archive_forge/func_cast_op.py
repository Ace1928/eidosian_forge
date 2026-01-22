import abc
import warnings
from functools import wraps
from typing import Tuple
import numpy as np
import cvxpy.settings as s
import cvxpy.utilities as u
import cvxpy.utilities.key_utils as ku
import cvxpy.utilities.performance_utils as perf
from cvxpy import error
from cvxpy.constraints import PSD, Equality, Inequality
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes
from cvxpy.utilities.shape import size_from_shape
@wraps(binary_op)
def cast_op(self, other):
    """A wrapped binary operator that can handle non-Expression arguments.
        """
    other = self.cast_to_const(other)
    return binary_op(self, other)