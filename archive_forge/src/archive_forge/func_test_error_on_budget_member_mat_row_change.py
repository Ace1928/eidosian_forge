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
def test_error_on_budget_member_mat_row_change(self):
    """
        Number of rows of budget membership mat is immutable.
        Hence, size of budget_rhs_vec is also immutable.
        """
    budget_mat = [[1, 0, 1], [0, 1, 0]]
    budget_rhs_vec = [1, 3]
    bu_set = BudgetSet(budget_mat, budget_rhs_vec)
    exc_str = ".*must have 2 rows to match shape of attribute 'budget_rhs_vec' \\(provided.*1 rows\\)"
    with self.assertRaisesRegex(ValueError, exc_str):
        bu_set.budget_membership_mat = [[1, 0, 1]]
    exc_str = ".*must have 2 entries to match shape of attribute 'budget_membership_mat' \\(provided.*1 entries\\)"
    with self.assertRaisesRegex(ValueError, exc_str):
        bu_set.budget_rhs_vec = [1]