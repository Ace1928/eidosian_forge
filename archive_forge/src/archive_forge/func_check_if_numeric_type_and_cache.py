import sys
import logging
from pyomo.common.deprecation import (
from pyomo.core.expr.expr_common import ExpressionType
from pyomo.core.expr.numeric_expr import NumericValue
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
@deprecated('check_if_numeric_type_and_cache() has been deprecated in favor of just calling as_numeric()', version='6.4.3')
def check_if_numeric_type_and_cache(obj):
    """Test if the argument is a numeric type by checking if we can add
    zero to it.  If that works, then we cache the value and return a
    NumericConstant object.

    """
    if check_if_numeric_type(obj):
        return as_numeric(obj)
    else:
        return obj