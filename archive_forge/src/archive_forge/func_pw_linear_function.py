from pyomo.common.autoslots import AutoSlots
from pyomo.core.expr.numeric_expr import NumericExpression
from weakref import ref as weakref_ref
@property
def pw_linear_function(self):
    return self._pw_linear_function()