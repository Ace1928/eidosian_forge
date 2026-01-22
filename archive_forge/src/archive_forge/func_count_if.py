import types
from itertools import islice
import logging
import traceback
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.deprecation import (
from .numvalue import (
from .base import ExpressionBase
from .boolean_value import BooleanValue, BooleanConstant
from .expr_common import _and, _or, _equiv, _inv, _xor, _impl, ExpressionType
from .numeric_expr import NumericExpression
import operator
def count_if(*args):
    """Creates a new CountIfExpression

    Counts the number of True-valued arguments

    Usage: count_if(m.Y1, m.Y2, ...)
    """
    return CountIfExpression(list(_flattened_boolean_args(args)))