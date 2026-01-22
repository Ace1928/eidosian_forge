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
def test_terminate_with_application_error(self):
    """
        Check that PyROS correctly raises ApplicationError
        in event of abnormal IPOPT termination.
        """
    m = ConcreteModel()
    m.p = Param(mutable=True, initialize=1.5)
    m.x1 = Var(initialize=-1)
    m.obj = Objective(expr=log(m.x1) * m.p)
    m.con = Constraint(expr=m.x1 * m.p >= -2)
    solver = SolverFactory('ipopt')
    solver.options['halt_on_ampl_error'] = 'yes'
    baron = SolverFactory('baron')
    box_set = BoxSet(bounds=[(1, 2)])
    pyros_solver = SolverFactory('pyros')
    with self.assertRaisesRegex(ApplicationError, 'Solver \\(ipopt\\) did not exit normally'):
        pyros_solver.solve(model=m, first_stage_variables=[m.x1], second_stage_variables=[], uncertain_params=[m.p], uncertainty_set=box_set, local_solver=solver, global_solver=baron, objective_focus=ObjectiveType.nominal, time_limit=1000)
    self.assertEqual(len(list(solver.options.keys())), 1, msg=f'Local subsolver {solver} options were changed by PyROS')
    self.assertEqual(solver.options['halt_on_ampl_error'], 'yes', msg=f"Local subsolver {solver} option 'halt_on_ampl_error' was changed by PyROS")
    self.assertEqual(len(list(baron.options.keys())), 0, msg=f'Global subsolver {baron} options were changed by PyROS')