from pyomo.core.base import (
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverResults
from pyomo.core.expr import value
from pyomo.core.base.set_types import NonNegativeIntegers, NonNegativeReals
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, MasterResult
from pyomo.opt.results import check_optimal_termination
from pyomo.core.expr.visitor import replace_expressions, identify_variables
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core import TransformationFactory
import itertools as it
import os
from copy import deepcopy
from pyomo.common.errors import ApplicationError
from pyomo.common.modeling import unique_component_name
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.pyros.util import TIC_TOC_SOLVE_TIME_ATTR, enforce_dr_degree
def construct_dr_polishing_problem(model_data, config):
    """
    Construct DR polishing problem from most recently added
    master problem.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    polishing_model : ConcreteModel
        Polishing model.

    Note
    ----
    Polishing problem is to minimize the L1-norm of the vector of
    all decision rule polynomial terms, subject to the original
    master problem constraints, with all first-stage variables
    (including epigraph) fixed. Optimality of the polished
    DR with respect to the master objective is also enforced.
    """
    master_model = model_data.master_model
    polishing_model = master_model.clone()
    nominal_polishing_block = polishing_model.scenarios[0, 0]
    decision_rule_var_set = ComponentSet((var for indexed_dr_var in nominal_polishing_block.util.decision_rule_vars for var in indexed_dr_var.values()))
    first_stage_vars = nominal_polishing_block.util.first_stage_variables
    for var in first_stage_vars:
        if var not in decision_rule_var_set:
            var.fix()
    if config.objective_focus == ObjectiveType.worst_case:
        polishing_model.zeta.fix()
    else:
        optimal_master_obj_value = value(polishing_model.obj)
        polishing_model.nominal_optimality_con = Constraint(expr=nominal_polishing_block.first_stage_objective + nominal_polishing_block.second_stage_objective <= optimal_master_obj_value)
    polishing_model.obj.deactivate()
    decision_rule_vars = nominal_polishing_block.util.decision_rule_vars
    nominal_polishing_block.util.polishing_vars = polishing_vars = []
    for idx, indexed_dr_var in enumerate(decision_rule_vars):
        indexed_polishing_var = Var(list(indexed_dr_var.keys()), domain=NonNegativeReals)
        nominal_polishing_block.add_component(unique_component_name(nominal_polishing_block, f'dr_polishing_var_{idx}'), indexed_polishing_var)
        polishing_vars.append(indexed_polishing_var)
    dr_eq_var_zip = zip(nominal_polishing_block.util.decision_rule_eqns, polishing_vars, nominal_polishing_block.util.second_stage_variables)
    nominal_polishing_block.util.polishing_abs_val_lb_cons = all_lb_cons = []
    nominal_polishing_block.util.polishing_abs_val_ub_cons = all_ub_cons = []
    for idx, (dr_eq, indexed_polishing_var, ss_var) in enumerate(dr_eq_var_zip):
        polishing_absolute_value_lb_cons = Constraint(indexed_polishing_var.index_set())
        polishing_absolute_value_ub_cons = Constraint(indexed_polishing_var.index_set())
        nominal_polishing_block.add_component(unique_component_name(polishing_model, f'polishing_abs_val_lb_con_{idx}'), polishing_absolute_value_lb_cons)
        nominal_polishing_block.add_component(unique_component_name(polishing_model, f'polishing_abs_val_ub_con_{idx}'), polishing_absolute_value_ub_cons)
        all_lb_cons.append(polishing_absolute_value_lb_cons)
        all_ub_cons.append(polishing_absolute_value_ub_cons)
        dr_expr_terms = dr_eq.body.args[:-1]
        for dr_eq_term in dr_expr_terms:
            dr_var_in_term = dr_eq_term.args[-1]
            dr_var_in_term_idx = dr_var_in_term.index()
            polishing_var = indexed_polishing_var[dr_var_in_term_idx]
            polishing_absolute_value_lb_cons[dr_var_in_term_idx] = -polishing_var - dr_eq_term <= 0
            polishing_absolute_value_ub_cons[dr_var_in_term_idx] = dr_eq_term - polishing_var <= 0
            if dr_var_in_term.fixed:
                polishing_var.fix()
                polishing_absolute_value_lb_cons[dr_var_in_term_idx].deactivate()
                polishing_absolute_value_ub_cons[dr_var_in_term_idx].deactivate()
            polishing_var.set_value(abs(value(dr_eq_term)))
    polishing_model.polishing_obj = Objective(expr=sum((sum(polishing_var.values()) for polishing_var in polishing_vars)))
    return polishing_model