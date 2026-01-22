from pyomo.gdp import GDP_Error, Disjunction
from pyomo.gdp.disjunct import _DisjunctData, Disjunct
import pyomo.core.expr as EXPR
from pyomo.core.base.component import _ComponentBase
from pyomo.core import (
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentMap, ComponentSet, OrderedSet
from pyomo.opt import TerminationCondition, SolverStatus
from weakref import ref as weakref_ref
from collections import defaultdict
import logging
def clone_without_expression_components(expr, substitute=None):
    """A function that is used to clone an expression.

    Cloning is roughly equivalent to calling ``copy.deepcopy``.
    However, the :attr:`clone_leaves` argument can be used to
    clone only interior (i.e. non-leaf) nodes in the expression
    tree.   Note that named expression objects are treated as
    leaves when :attr:`clone_leaves` is :const:`True`, and hence
    those subexpressions are not cloned.

    This function uses a non-recursive
    logic, which makes it more scalable than the logic in
    ``copy.deepcopy``.

    Args:
        expr: The expression that will be cloned.
        substitute (dict): A dictionary mapping object ids to
            objects.  This dictionary has the same semantics as
            the memo object used with ``copy.deepcopy``.  Defaults
            to None, which indicates that no user-defined
            dictionary is used.

    Returns:
        The cloned expression.
    """
    if substitute is None:
        substitute = {}
    visitor = EXPR.ExpressionReplacementVisitor(substitute=substitute, remove_named_expressions=True)
    return visitor.walk_expression(expr)