import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
def substitute_pyomo2casadi(expr, templatemap):
    """Substitute IndexTemplates in an expression tree.

    This substitution function is used to replace Pyomo intrinsic
    functions with CasADi functions.

    Args:
        expr: a Pyomo expression
        templatemap: dictionary mapping _GetItemIndexer objects to
            mutable params

    Returns:
        a new expression tree with all substitutions done
    """
    if not casadi_available:
        raise DAE_Error('CASADI is not installed.  Cannot substitute CasADi variables and intrinsic functions.')
    visitor = Substitute_Pyomo2Casadi_Visitor(templatemap)
    return visitor.walk_expression(expr)