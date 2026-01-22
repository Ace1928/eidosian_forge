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
@deprecated('ExpressionReplacementVisitor: this walker has been ported to derive from StreamBasedExpressionVisitor.  dfs_postorder_stack() has been replaced with walk_expression()', version='6.2')
def dfs_postorder_stack(self, expr):
    return self.walk_expression(expr)