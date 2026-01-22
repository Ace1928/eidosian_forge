from contextlib import contextmanager
from math import fabs
from pyomo.common.collections import ComponentMap
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.solve_discrete_problem import solve_MILP_discrete_problem
from pyomo.contrib.gdpopt.util import _DoNothing
from pyomo.core import Block, Constraint, Objective, Var, maximize, value
from pyomo.gdp import Disjunct
from pyomo.opt import TerminationCondition as tc
@contextmanager
def preserve_discrete_problem_feasible_region(discrete_problem_util_block, config, original_bounds=None):
    if config.mip_presolve and original_bounds is None:
        original_bounds = _collect_original_bounds(discrete_problem_util_block)
    yield
    if config.mip_presolve:
        _restore_bounds(original_bounds)