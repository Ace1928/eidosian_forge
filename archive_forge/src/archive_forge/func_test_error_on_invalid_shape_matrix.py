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
def test_error_on_invalid_shape_matrix(self):
    """
        Test exceptional cases of invalid square shape matrix
        arguments
        """
    center = [0, 0]
    scale = 3
    with self.assertRaisesRegex(ValueError, 'Shape matrix must be symmetric', msg='Asymmetric shape matrix test failed'):
        EllipsoidalSet(center, [[1, 1], [0, 1]], scale)
    with self.assertRaises(np.linalg.LinAlgError, msg='Singular shape matrix test failed'):
        EllipsoidalSet(center, [[0, 0], [0, 0]], scale)
    with self.assertRaisesRegex(ValueError, 'Non positive-definite.*', msg='Indefinite shape matrix test failed'):
        EllipsoidalSet(center, [[1, 0], [0, -2]], scale)
    eset = EllipsoidalSet(center, [[1, 0], [0, 2]], scale)
    with self.assertRaisesRegex(ValueError, 'Shape matrix must be symmetric', msg='Asymmetric shape matrix test failed'):
        eset.shape_matrix = [[1, 1], [0, 1]]
    with self.assertRaises(np.linalg.LinAlgError, msg='Singular shape matrix test failed'):
        eset.shape_matrix = [[0, 0], [0, 0]]
    with self.assertRaisesRegex(ValueError, 'Non positive-definite.*', msg='Indefinite shape matrix test failed'):
        eset.shape_matrix = [[1, 0], [0, -2]]