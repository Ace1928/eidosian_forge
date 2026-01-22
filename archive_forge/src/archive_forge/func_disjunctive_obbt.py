from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt, BoundsManager
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.core.expr import identify_variables
from pyomo.core import Constraint, Objective, TransformationFactory, minimize, value
from pyomo.opt import SolverFactory
from pyomo.gdp.disjunct import Disjunct
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.opt import TerminationCondition as tc
def disjunctive_obbt(model, solver):
    """Provides Optimality-based bounds tightening to a model using a solver."""
    model._disjuncts_to_process = list(model.component_data_objects(ctype=Disjunct, active=True, descend_into=(Block, Disjunct), descent_order=TraversalStrategy.BreadthFirstSearch))
    if model.ctype == Disjunct:
        model._disjuncts_to_process.insert(0, model)
    linear_var_set = ComponentSet()
    for constr in model.component_data_objects(Constraint, active=True, descend_into=(Block, Disjunct)):
        if constr.body.polynomial_degree() in linear_degrees:
            linear_var_set.update(identify_variables(constr.body, include_fixed=False))
    model._disj_bnds_linear_vars = list(linear_var_set)
    for disj_idx, disjunct in enumerate(model._disjuncts_to_process):
        var_bnds = obbt_disjunct(model, disj_idx, solver)
        if var_bnds is not None:
            if not hasattr(disjunct, '_disj_var_bounds'):
                disjunct._disj_var_bounds = var_bnds
            else:
                for var, new_bnds in var_bnds.items():
                    old_lb, old_ub = disjunct._disj_var_bounds.get(var, (-inf, inf))
                    new_lb, new_ub = new_bnds
                    disjunct._disj_var_bounds[var] = (max(old_lb, new_lb), min(old_ub, new_ub))
        else:
            disjunct.deactivate()