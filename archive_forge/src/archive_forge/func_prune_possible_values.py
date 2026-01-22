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
def prune_possible_values(block_scope, possible_values, config):
    top_level_scope = block_scope.model()
    tmp_name = unique_component_name(top_level_scope, '_induced_linearity_prune_data')
    tmp_orig_blk = Block()
    setattr(top_level_scope, tmp_name, tmp_orig_blk)
    tmp_orig_blk._possible_values = possible_values
    tmp_orig_blk._possible_value_vars = list((v for v in possible_values))
    tmp_orig_blk._tmp_block_scope = (block_scope,)
    model = top_level_scope.clone()
    tmp_clone_blk = getattr(model, tmp_name)
    for obj in model.component_data_objects(Objective, active=True):
        obj.deactivate()
    for constr in model.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)):
        if constr.body.polynomial_degree() not in (1, 0):
            constr.deactivate()
    if block_scope.ctype == Disjunct:
        disj = tmp_clone_blk._tmp_block_scope[0]
        disj.indicator_var.fix(1)
        TransformationFactory('gdp.bigm').apply_to(model)
    for d in model.component_data_objects(Disjunction):
        d.deactivate()
    for d in model.component_data_objects(Disjunct):
        d._deactivate_without_fixing_indicator()
    tmp_clone_blk.test_feasible = Constraint()
    tmp_clone_blk._obj = Objective(expr=1)
    for eff_discr_var, vals in tmp_clone_blk._possible_values.items():
        val_feasible = {}
        for val in vals:
            tmp_clone_blk.test_feasible.set_value(eff_discr_var == val)
            with SuppressConstantObjectiveWarning():
                res = SolverFactory(config.pruning_solver).solve(model)
            if res.solver.termination_condition is tc.infeasible:
                val_feasible[val] = False
        tmp_clone_blk._possible_values[eff_discr_var] = set((v for v in tmp_clone_blk._possible_values[eff_discr_var] if val_feasible.get(v, True)))
    for i, var in enumerate(tmp_orig_blk._possible_value_vars):
        possible_values[var] = tmp_clone_blk._possible_values[tmp_clone_blk._possible_value_vars[i]]
    return possible_values