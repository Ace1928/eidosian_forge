from contextlib import contextmanager
from math import fabs
from pyomo.common.collections import ComponentMap
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.solve_discrete_problem import solve_MILP_discrete_problem
from pyomo.contrib.gdpopt.util import _DoNothing
from pyomo.core import Block, Constraint, Objective, Var, maximize, value
from pyomo.gdp import Disjunct
from pyomo.opt import TerminationCondition as tc
def init_fixed_disjuncts(util_block, discrete_problem_util_block, subprob_util_block, config, solver):
    """Initialize by solving the problem with the current disjunct values."""
    config.logger.info('Generating initial linear GDP approximation by solving subproblem with original user-specified disjunct values.')
    solver._log_header(config.logger)
    with preserve_discrete_problem_feasible_region(discrete_problem_util_block, config):
        already_fixed = set()
        for disj in discrete_problem_util_block.disjunct_list:
            indicator = disj.indicator_var
            if indicator.fixed:
                already_fixed.add(disj)
            else:
                indicator.fix()
        mip_termination = solve_MILP_discrete_problem(discrete_problem_util_block, solver, config)
        for disj in discrete_problem_util_block.disjunct_list:
            if disj not in already_fixed:
                disj.indicator_var.unfix()
    if mip_termination is not tc.infeasible:
        solver._fix_discrete_soln_solve_subproblem_and_add_cuts(discrete_problem_util_block, subprob_util_block, config)
        add_no_good_cut(discrete_problem_util_block, config)
    else:
        config.logger.error('MILP relaxation infeasible for initial user-specified disjunct values. Skipping initialization.')
    solver.initialization_iteration += 1