import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def convert_temp_R_to_F(self, value_in_R):
    """
        Convert a value in Rankine to degrees Fahrenheit.  Note that
        this method converts a numerical value only. If you need
        temperature conversions in expressions, please work in
        absolute temperatures only.
        """
    return self._pint_convert_temp_from_to(value_in_R, self._pint_registry.rankine, self._pint_registry.degF)