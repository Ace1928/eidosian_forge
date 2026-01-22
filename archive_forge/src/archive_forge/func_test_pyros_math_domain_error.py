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
def test_pyros_math_domain_error(self):
    """
        Test PyROS on a two-stage problem, discrete
        set type with a math domain error evaluating
        performance constraint expressions in separation.
        """
    m = ConcreteModel()
    m.q = Param(initialize=1, mutable=True)
    m.x1 = Var(initialize=1, bounds=(0, 1))
    m.x2 = Var(initialize=2, bounds=(-m.q, log(m.q)))
    m.obj = Objective(expr=m.x1 + m.x2)
    box_set = BoxSet(bounds=[[0, 1]])
    local_solver = SolverFactory('baron')
    global_solver = SolverFactory('baron')
    pyros_solver = SolverFactory('pyros')
    with self.assertRaisesRegex(expected_exception=ArithmeticError, expected_regex='Evaluation of performance constraint.*math domain error.*', msg='ValueError arising from math domain error not raised'):
        pyros_solver.solve(model=m, first_stage_variables=[m.x1], second_stage_variables=[m.x2], uncertain_params=[m.q], uncertainty_set=box_set, local_solver=local_solver, global_solver=global_solver, decision_rule_order=1, tee=True)