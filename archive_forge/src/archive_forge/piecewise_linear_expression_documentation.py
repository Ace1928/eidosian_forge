from pyomo.common.autoslots import AutoSlots
from pyomo.core.expr.numeric_expr import NumericExpression
from weakref import ref as weakref_ref

    A numeric expression node representing a specific instantiation of a
    PiecewiseLinearFunction.

    Args:
        args (list or tuple): Children of this node
        pw_linear_function (PiecewiseLinearFunction): piece-wise linear function
            of which this node is an instance.
    