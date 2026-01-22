import logging
from pyomo.common.collections import ComponentMap
from pyomo.core import (
from pyomo.repn import generate_standard_repn
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
import pyomo.core.expr as EXPR
from pyomo.opt import ProblemSense
from pyomo.contrib.gdpopt.util import get_main_elapsed_time, time_code
from pyomo.util.model_size import build_model_size_report
from pyomo.common.dependencies import attempt_import
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
import math
def copy_var_list_values_from_solution_pool(from_list, to_list, config, solver_model, var_map, solution_name, ignore_integrality=False):
    """Copy variable values from the solution pool to another list.

    Parameters
    ----------
    from_list : list
        The variables that provides the values to copy from.
    to_list : list
        The variables that need to set value.
    config : ConfigBlock
        The specific configurations for MindtPy.
    solver_model : solver model
        The solver model derived from pyomo model.
    var_map : dict
        The map of pyomo variables to solver variables.
    solution_name : int or str
        The name of the solution in the solution pool.
    ignore_integrality : bool, optional
        Whether to ignore the integrality of integer variables, by default False.
    """
    for v_from, v_to in zip(from_list, to_list):
        if config.mip_solver == 'cplex_persistent':
            var_val = solver_model.solution.pool.get_values(solution_name, var_map[v_from])
        elif config.mip_solver == 'gurobi_persistent':
            solver_model.setParam(gurobipy.GRB.Param.SolutionNumber, solution_name)
            var_val = var_map[v_from].Xn
        set_var_valid_value(v_to, var_val, config.integer_tolerance, config.zero_tolerance, ignore_integrality)