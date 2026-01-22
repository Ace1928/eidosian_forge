import logging
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.trustregion.util import minIgnoreNone, maxIgnoreNone
from pyomo.core import (
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.visitor import identify_variables, ExpressionReplacementVisitor
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types
from pyomo.opt import SolverFactory, check_optimal_termination
def updateDecisionVariableBounds(self, radius):
    """
        Update the TRSP_k decision variable bounds

        This corresponds to:
            || E^{-1} (u - u_k) || <= trust_radius
        We omit E^{-1} because we assume that the users have correctly scaled
        their variables.
        """
    for var in self.decision_variables:
        var.setlb(maxIgnoreNone(value(var) - radius, self.initial_decision_bounds[var.name][0]))
        var.setub(minIgnoreNone(value(var) + radius, self.initial_decision_bounds[var.name][1]))