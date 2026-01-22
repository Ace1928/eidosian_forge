import operator
from pyomo.common.deprecation import deprecated
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.numeric_types import (
from .base import ExpressionBase
from .boolean_value import BooleanValue
from .expr_common import _lt, _le, _eq, ExpressionType
from .numvalue import is_potentially_variable, is_constant
from .visitor import polynomial_degree
def tuple_to_relational_expr(args):
    if len(args) == 2:
        return EqualityExpression(args)
    else:
        return inequality(*args)