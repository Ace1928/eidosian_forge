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
@unittest.skipUnless(baron_available, 'Global NLP solver is not available.')
def test_add_bounds_on_uncertain_parameters(self):
    m = ConcreteModel()
    m.util = Block()
    m.util.uncertain_param_vars = Var([0, 1], initialize=0.5)
    bounds = [(-1, 1), (-1, 1)]
    Q1 = BoxSet(bounds=bounds)
    Q2 = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[5, 5])
    Q = IntersectionSet(Q1=Q1, Q2=Q2)
    config = Block()
    config.uncertainty_set = Q
    config.global_solver = SolverFactory('baron')
    IntersectionSet.add_bounds_on_uncertain_parameters(m, config)
    self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, 'Bounds not added correctly for IntersectionSet')
    self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, 'Bounds not added correctly for IntersectionSet')
    self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, 'Bounds not added correctly for IntersectionSet')
    self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, 'Bounds not added correctly for IntersectionSet')