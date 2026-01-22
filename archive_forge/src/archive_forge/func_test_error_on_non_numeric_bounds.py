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
def test_error_on_non_numeric_bounds(self):
    """
        Test that ValueError is raised if box set bounds
        are set to array-like with entries of a non-numeric
        type (such as int, float).
        """
    new_bounds = [[1, 'test'], [3, 2]]
    exc_str = "Entry 'test' of the argument `bounds` is not a valid numeric type \\(provided type 'str'\\)"
    with self.assertRaisesRegex(TypeError, exc_str):
        BoxSet(new_bounds)
    bset = BoxSet(bounds=[[1, 2], [3, 4]])
    with self.assertRaisesRegex(TypeError, exc_str):
        bset.bounds = new_bounds