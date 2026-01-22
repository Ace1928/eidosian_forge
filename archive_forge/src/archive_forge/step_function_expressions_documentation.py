from pyomo.contrib.cp.interval_var import (
from pyomo.core.base.component import Component
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.logical_expr import BooleanExpression

    An expression representing the constraint that a cumulative function is
    required to take values within a tuple of bounds over a specified time
    interval. (Often used to enforce limits on resource availability.)

    Args:
        cumul_func (CumulativeFunction): Step function being constrained
        bounds (tuple of two integers): Lower and upper bounds to enforce on
            the cumulative function
        times (tuple of two integers): The time interval (start, end) over
            which to enforce the bounds on the values of the cumulative
            function.
    