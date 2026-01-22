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
def atleast(n, *args):
    """Creates a new AtLeastExpression

    Require at least n arguments to be True, to make the expression True

    Usage: atleast(2, m.Y1, m.Y2, m.Y3, ...)

    """
    result = AtLeastExpression([n] + list(_flattened_boolean_args(args)))
    return result