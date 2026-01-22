import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
def convert_pyomo2scipy(expr, templatemap):
    """Substitute _GetItem nodes in an expression tree.

    This substitution function is used to replace Pyomo _GetItem
    nodes with mutable Params.

    Args:
        templatemap: dictionary mapping _GetItemIndexer objects to
            mutable params

    Returns:
        a new expression tree with all substitutions done
    """
    if not scipy_available:
        raise DAE_Error('SciPy is not installed. Cannot substitute SciPy intrinsic functions.')
    visitor = Pyomo2Scipy_Visitor(templatemap)
    return visitor.walk_expression(expr)