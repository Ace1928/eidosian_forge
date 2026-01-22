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
def walk_expression_nonrecursive(self, expr):
    """Nonrecursively walk an expression, calling registered callbacks.

        This routine is safer than the recursive walkers for deep (or
        unbalanced) trees.  It is, however, slightly slower than the
        recursive implementations.

        """
    if self.initializeWalker is not None:
        walk, result = self.initializeWalker(expr)
        if not walk:
            return result
        elif result is not None:
            expr = result
    if self.enterNode is not None:
        tmp = self.enterNode(expr)
        if tmp is None:
            args = data = None
        else:
            args, data = tmp
    else:
        args = None
        data = []
    if args is None:
        if type(expr) in nonpyomo_leaf_types or not expr.is_expression_type():
            args = ()
        else:
            args = expr.args
    if hasattr(args, '__enter__'):
        args.__enter__()
    node = expr
    return self._nonrecursive_walker_loop((None, node, args, len(args) - 1, data, -1))