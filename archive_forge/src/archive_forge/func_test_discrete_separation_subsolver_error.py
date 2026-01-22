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
@unittest.expectedFailure
def test_discrete_separation_subsolver_error(self):
    """
        Test PyROS for two-stage problem with discrete type set,
        subsolver error status.
        """
    m = ConcreteModel()
    m.q = Param(initialize=1, mutable=True)
    m.x1 = Var(initialize=1, bounds=(0, 1))
    m.x2 = Var(initialize=2, bounds=(None, log(m.q)))
    m.obj = Objective(expr=m.x1 + m.x2, sense=maximize)
    discrete_set = DiscreteScenarioSet(scenarios=[(1,), (0,)])
    local_solver = SolverFactory('ipopt')
    global_solver = SolverFactory('baron')
    pyros_solver = SolverFactory('pyros')
    res = pyros_solver.solve(model=m, first_stage_variables=[m.x1], second_stage_variables=[m.x2], uncertain_params=[m.q], uncertainty_set=discrete_set, local_solver=local_solver, global_solver=global_solver, decision_rule_order=1, tee=True)
    self.assertEqual(res.pyros_termination_condition, pyrosTerminationCondition.subsolver_error, msg=f'Returned termination condition for separation errortest is not {pyrosTerminationCondition.subsolver_error}.')