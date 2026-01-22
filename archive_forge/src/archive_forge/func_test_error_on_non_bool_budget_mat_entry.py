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
def test_error_on_non_bool_budget_mat_entry(self):
    """
        Test ValueError raised if budget membership mat has
        entry which is not a 0-1 value.
        """
    invalid_budget_mat = [[1, 0, 1], [1, 1, 0.1]]
    budget_rhs_vec = [1, 1]
    exc_str = 'Attempting.*entries.*not 0-1 values \\(example: 0.1\\).*'
    with self.assertRaisesRegex(ValueError, exc_str):
        BudgetSet(invalid_budget_mat, budget_rhs_vec)
    buset = BudgetSet([[1, 0, 1], [1, 1, 0]], budget_rhs_vec)
    with self.assertRaisesRegex(ValueError, exc_str):
        buset.budget_membership_mat = invalid_budget_mat