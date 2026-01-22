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
def test_coefficient_matching_raises_error_4_3(self):
    """
        Check that result for instance with constraint subject to
        coefficient matching results in exception certifying robustness
        cannot be certified where expected. Model
        is based on Mitsos (2011) semi-infinite programming instance
        4_3.
        """
    m = self.create_mitsos_4_3()
    baron = SolverFactory('baron')
    pyros_solver = SolverFactory('pyros')
    dr_orders = [1, 2]
    for dr_order in dr_orders:
        regex_assert_mgr = self.assertRaisesRegex(ValueError, expected_regex='Coefficient matching unsuccessful. See the solver logs.')
        logging_intercept_mgr = LoggingIntercept(level=logging.ERROR)
        with regex_assert_mgr, logging_intercept_mgr as LOG:
            pyros_solver.solve(model=m, first_stage_variables=[], second_stage_variables=[m.x1, m.x2, m.x3], uncertain_params=[m.u], uncertainty_set=BoxSet(bounds=[[0, 1]]), local_solver=baron, global_solver=baron, objective_focus=ObjectiveType.worst_case, decision_rule_order=dr_order, solve_master_globally=True, bypass_local_separation=True, robust_feasibility_tolerance=0.0001)
        detailed_error_msg = LOG.getvalue()
        self.assertRegex(detailed_error_msg[:-1], 'Equality constraint.*cannot be guaranteed to be robustly feasible.*Consider editing this constraint.*')