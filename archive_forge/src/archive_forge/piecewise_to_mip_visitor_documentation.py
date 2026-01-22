from pyomo.contrib.piecewise.piecewise_linear_expression import (
from pyomo.core import Expression
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor

    Expression walker to replace PiecewiseLinearExpressions when creating
    equivalent MIP formulations.

    Args:
        transform_pw_linear_expression (function): a callback that accepts
            a PiecewiseLinearExpression, its parent PiecewiseLinearFunction,
            and a transformation Block. It is expected to convert the
            PiecewiseLinearExpression to MIP form, and return the Var (or
            other expression) that will replace it in the expression.
        transBlock (Block): transformation Block to pass to the above
            callback
    