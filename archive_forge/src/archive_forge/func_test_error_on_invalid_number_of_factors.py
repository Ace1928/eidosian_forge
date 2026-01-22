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
def test_error_on_invalid_number_of_factors(self):
    """
        Test ValueError raised if number of factors
        is negative int, or AttributeError
        if attempting to update (should be immutable).
        """
    exc_str = ".*'number_of_factors' must be a positive int \\(provided value -1\\)"
    with self.assertRaisesRegex(ValueError, exc_str):
        FactorModelSet(origin=[0], number_of_factors=-1, psi_mat=[[1, 1]], beta=0.1)
    fset = FactorModelSet(origin=[0], number_of_factors=2, psi_mat=[[1, 1]], beta=0.1)
    exc_str = ".*'number_of_factors' is immutable"
    with self.assertRaisesRegex(AttributeError, exc_str):
        fset.number_of_factors = 3