import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
import math
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import (
from pyomo.contrib.pyros.master_problem_methods import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
from pyomo.environ import (
import logging
from itertools import chain
def regression_test_constant_drs(self):
    model = m = ConcreteModel()
    m.name = 's381'
    m.x1 = Var(within=Reals, bounds=(0, None), initialize=0.1)
    m.x2 = Var(within=Reals, bounds=(0, None), initialize=0.1)
    m.x3 = Var(within=Reals, bounds=(0, None), initialize=0.1)
    m.decision_vars = [m.x1, m.x2, m.x3]
    m.set_params = Set(initialize=list(range(4)))
    m.p = Param(m.set_params, initialize=2, mutable=True)
    m.uncertain_params = [m.p]
    m.obj = Objective(expr=(m.x1 - 1) * 2, sense=minimize)
    m.con1 = Constraint(expr=m.p[1] * m.x1 + m.x2 + m.x3 <= 2)
    box_set = BoxSet(bounds=[(1.8, 2.2)])
    solver = SolverFactory('baron')
    pyros = SolverFactory('pyros')
    results = pyros.solve(model=m, first_stage_variables=m.decision_vars, second_stage_variables=[], uncertain_params=[m.p[1]], uncertainty_set=box_set, local_solver=solver, global_solver=solver, options={'objective_focus': ObjectiveType.nominal})
    self.assertTrue(results.pyros_termination_condition, pyrosTerminationCondition.robust_feasible)