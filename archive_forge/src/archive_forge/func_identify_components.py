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
def identify_components(expr, component_types):
    """
    A generator that yields a sequence of nodes
    in an expression tree that belong to a specified set.

    Args:
        expr: The root node of an expression tree.
        component_types (set or list): A set of class
            types that will be matched during the search.

    Yields:
        Each node that is found.
    """
    visitor = _ComponentVisitor(component_types)
    yield from visitor.xbfs_yield_leaves(expr)