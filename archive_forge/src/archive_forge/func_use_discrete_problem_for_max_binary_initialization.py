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
def use_discrete_problem_for_max_binary_initialization(discrete_problem_util_block):
    m = discrete_problem_util_block.parent_block()
    original_objective = next(m.component_data_objects(Objective, active=True, descend_into=True))
    original_objective.deactivate()
    binary_vars = (v for v in m.component_data_objects(ctype=Var, descend_into=(Block, Disjunct)) if v.is_binary() and (not v.fixed))
    discrete_problem_util_block.max_binary_obj = Objective(expr=sum(binary_vars), sense=maximize)
    yield
    del discrete_problem_util_block.max_binary_obj
    original_objective.activate()