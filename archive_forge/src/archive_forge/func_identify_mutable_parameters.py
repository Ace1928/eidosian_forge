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
def identify_mutable_parameters(expr):
    """
    A generator that yields a sequence of mutable
    parameters in an expression tree.

    Args:
        expr: The root node of an expression tree.

    Yields:
        Each mutable parameter that is found.
    """
    visitor = _MutableParamVisitor()
    yield from visitor.xbfs_yield_leaves(expr)