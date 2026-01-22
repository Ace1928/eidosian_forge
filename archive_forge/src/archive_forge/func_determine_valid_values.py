import logging
import textwrap
from math import fabs
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.preprocessing.util import SuppressConstantObjectiveWarning
from pyomo.core import (
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
def determine_valid_values(block, discr_var_to_constrs_map, config):
    """Calculate valid values for each effectively discrete variable.

    We need the set of possible values for the effectively discrete variable in
    order to do the reformulations.

    Right now, we select a naive approach where we look for variables in the
    discreteness-inducing constraints. We then adjust their values and see if
    things are still feasible. Based on their coefficient values, we can infer a
    set of allowable values for the effectively discrete variable.

    Args:
        block: The model or a disjunct on the model.

    """
    possible_values = ComponentMap()
    for eff_discr_var, constrs in discr_var_to_constrs_map.items():
        for constr in constrs:
            repn = generate_standard_repn(constr.body)
            var_coef = sum((coef for i, coef in enumerate(repn.linear_coefs) if repn.linear_vars[i] is eff_discr_var))
            const = -(repn.constant - constr.upper) / var_coef
            possible_vals = set((const,))
            for i, var in enumerate(repn.linear_vars):
                if var is eff_discr_var:
                    continue
                coef = -repn.linear_coefs[i] / var_coef
                if var.is_binary():
                    var_values = (0, coef)
                elif var.is_integer():
                    var_values = [v * coef for v in range(var.lb, var.ub + 1)]
                else:
                    raise ValueError('%s has unacceptable variable domain: %s' % (var.name, var.domain))
                possible_vals = set((v1 + v2 for v1 in possible_vals for v2 in var_values))
            old_possible_vals = possible_values.get(eff_discr_var, None)
            if old_possible_vals is not None:
                possible_values[eff_discr_var] = old_possible_vals & possible_vals
            else:
                possible_values[eff_discr_var] = possible_vals
    possible_values = prune_possible_values(block, possible_values, config)
    return possible_values