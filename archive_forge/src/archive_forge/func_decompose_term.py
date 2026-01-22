import collections
import enum
import logging
import math
import operator
from pyomo.common.dependencies import attempt_import
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_common import (
from pyomo.core.expr.base import ExpressionBase, NPV_Mixin, visitor
def decompose_term(expr):
    """A function that returns a tuple consisting of (1) a flag indicating
    whether the expression is linear, and (2) a list of tuples that
    represents the terms in the linear expression.

    Args:
        expr (expression): The root node of an expression tree

    Returns:
        A tuple with the form ``(flag, list)``.  If :attr:`flag` is
        :const:`False`, then a nonlinear term has been found, and
        :const:`list` is :const:`None`.  Otherwise, :const:`list` is a
        list of tuples: ``(coef, value)``.  If :attr:`value` is
        :const:`None`, then this represents a constant term with value
        :attr:`coef`.  Otherwise, :attr:`value` is a variable object,
        and :attr:`coef` is the numeric coefficient.

    """
    if expr.__class__ in nonpyomo_leaf_types or not expr.is_potentially_variable():
        return (True, [(expr, None)])
    elif expr.is_variable_type():
        return (True, [(1, expr)])
    else:
        try:
            terms = [t_ for t_ in _decompose_linear_terms(expr)]
            return (True, terms)
        except LinearDecompositionError:
            return (False, None)