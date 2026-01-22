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
@unittest.skipUnless(baron_license_is_valid and scip_available and scip_license_is_valid, 'Global solvers BARON and SCIP not both available and licensed')
def test_coeff_matching_solver_insensitive(self):
    """
        Check that result for instance with constraint subject to
        coefficient matching is insensitive to subsolver settings. Based
        on Mitsos (2011) semi-infinite programming instance 4_3.
        """
    m = self.create_mitsos_4_3()
    baron = SolverFactory('baron')
    scip = SolverFactory('scip')
    pyros_solver = SolverFactory('pyros')
    solver_names = {'baron': baron, 'scip': scip}
    for name, solver in solver_names.items():
        res = pyros_solver.solve(model=m, first_stage_variables=[], second_stage_variables=[m.x1, m.x2, m.x3], uncertain_params=[m.u], uncertainty_set=BoxSet(bounds=[[0, 1]]), local_solver=solver, global_solver=solver, objective_focus=ObjectiveType.worst_case, solve_master_globally=True, bypass_local_separation=True, robust_feasibility_tolerance=0.0001)
        self.assertEqual(first=res.iterations, second=2, msg=f'Iterations for Watson 43 instance solved with subsolver {name} not as expected')
        np.testing.assert_allclose(actual=res.final_objective_value, desired=0.9781633, rtol=0, atol=0.005, err_msg=f'Final objective for Watson 43 instance solved with subsolver {name} not as expected')