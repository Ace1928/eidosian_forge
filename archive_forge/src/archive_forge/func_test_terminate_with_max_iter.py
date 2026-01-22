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
@unittest.skipUnless(baron_license_is_valid, 'Global NLP solver is not available and licensed.')
@unittest.skipUnless(baron_version < (23, 1, 5) or baron_version >= (23, 6, 23), 'Test known to fail for BARON 23.1.5 and versions preceding 23.6.23')
def test_terminate_with_max_iter(self):
    m = ConcreteModel()
    m.x1 = Var(initialize=0, bounds=(0, None))
    m.x2 = Var(initialize=0, bounds=(0, None))
    m.x3 = Var(initialize=0, bounds=(None, None))
    m.u = Param(initialize=1.125, mutable=True)
    m.con1 = Constraint(expr=m.x1 * m.u ** 0.5 - m.x2 * m.u <= 2)
    m.con2 = Constraint(expr=m.x1 ** 2 - m.x2 ** 2 * m.u == m.x3)
    m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - 1) ** 2)
    interval = BoxSet(bounds=[(0.25, 2)])
    pyros_solver = SolverFactory('pyros')
    local_subsolver = SolverFactory('baron')
    global_subsolver = SolverFactory('baron')
    results = pyros_solver.solve(model=m, first_stage_variables=[m.x1], second_stage_variables=[m.x2], uncertain_params=[m.u], uncertainty_set=interval, local_solver=local_subsolver, global_solver=global_subsolver, options={'objective_focus': ObjectiveType.worst_case, 'solve_master_globally': True, 'max_iter': 1, 'decision_rule_order': 2})
    self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.max_iter, msg='Returned termination condition is not return max_iter.')
    self.assertEqual(results.iterations, 1, msg=f'Number of iterations in results object is {results.iterations}, but expected value 1.')