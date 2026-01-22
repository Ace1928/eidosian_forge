import collections
from .visitor import StreamBasedExpressionVisitor
from .numvalue import nonpyomo_leaf_types
from pyomo.core.expr import (
from typing import List
from pyomo.common.collections import Sequence
from pyomo.common.errors import PyomoException
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import native_types
def convert_expression_to_prefix_notation(expr, include_named_exprs=True):
    """
    This function converts pyomo expressions to a list that looks very
    much like prefix notation.  The result can be used in equality
    comparisons to compare expression trees.

    Note that the data structure returned by this function might be
    changed in the future. However, we will maintain that the result
    can be used in equality comparisons.

    Also note that the result should really only be used in equality
    comparisons if the equality comparison is expected to return
    True. If the expressions being compared are expected to be
    different, then the equality comparison will often result in an
    error rather than returning False.

    m = ConcreteModel()
    m.x = Var()
    m.y = Var()

    e1 = m.x * m.y
    e2 = m.x * m.y
    e3 = m.x + m.y

    convert_expression_to_prefix_notation(e1) == convert_expression_to_prefix_notation(e2)  # True
    convert_expression_to_prefix_notation(e1) == convert_expression_to_prefix_notation(e3)  # Error

    However, the compare_expressions function can be used:

    compare_expressions(e1, e2)  # True
    compare_expressions(e1, e3)  # False

    Parameters
    ----------
    expr: NumericValue
        A Pyomo expression, Var, or Param

    Returns
    -------
    prefix_notation: list
        The expression in prefix notation

    """
    visitor = PrefixVisitor(include_named_exprs=include_named_exprs)
    if isinstance(expr, Sequence):
        return expr.__class__((visitor.walk_expression(e) for e in expr))
    else:
        return visitor.walk_expression(expr)