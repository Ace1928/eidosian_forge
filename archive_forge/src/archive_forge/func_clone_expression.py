import inspect
import logging
import sys
from copy import deepcopy
from collections import deque
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import DeveloperError, TemplateExpressionError
from pyomo.common.numeric_types import (
import pyomo.core.expr.expr_common as common
from pyomo.core.expr.symbol_map import SymbolMap
def clone_expression(expr, substitute=None):
    """A function that is used to clone an expression.

    Cloning is equivalent to calling ``copy.deepcopy`` with no Block
    scope.  That is, the expression tree is duplicated, but no Pyomo
    components (leaf nodes *or* named Expressions) are duplicated.

    Args:
        expr: The expression that will be cloned.
        substitute (dict): A dictionary mapping object ids to
            objects. This dictionary has the same semantics as
            the memo object used with ``copy.deepcopy``. Defaults
            to None, which indicates that no user-defined
            dictionary is used.

    Returns:
        The cloned expression.

    """
    common.clone_counter._count += 1
    memo = {'__block_scope__': {id(None): False}}
    if substitute:
        expr = replace_expressions(expr, substitute)
    return deepcopy(expr, memo)