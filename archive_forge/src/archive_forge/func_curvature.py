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
@property
def curvature(self) -> str:
    """str : The curvature of the expression.
        """
    if self.is_constant():
        curvature_str = s.CONSTANT
    elif self.is_affine():
        curvature_str = s.AFFINE
    elif self.is_convex():
        curvature_str = s.CONVEX
    elif self.is_concave():
        curvature_str = s.CONCAVE
    elif self.is_log_log_affine():
        curvature_str = s.LOG_LOG_AFFINE
    elif self.is_log_log_convex():
        curvature_str = s.LOG_LOG_CONVEX
    elif self.is_log_log_concave():
        curvature_str = s.LOG_LOG_CONCAVE
    elif self.is_quasilinear():
        curvature_str = s.QUASILINEAR
    elif self.is_quasiconvex():
        curvature_str = s.QUASICONVEX
    elif self.is_quasiconcave():
        curvature_str = s.QUASICONCAVE
    else:
        curvature_str = s.UNKNOWN
    return curvature_str