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
def test_set_with_zero_half_lengths(self):
    half_lengths = [1, 0, 2, 0]
    center = [1, 1, 1, 1]
    ell = AxisAlignedEllipsoidalSet(center, half_lengths)
    m = ConcreteModel()
    m.v1 = Var()
    m.v2 = Var([1, 2])
    m.v3 = Var()
    conlist = ell.set_as_constraint([m.v1, m.v2, m.v3])
    eq_cons = [con for con in conlist.values() if con.equality]
    self.assertEqual(len(conlist), 3, msg=f'Constraint list for this `AxisAlignedEllipsoidalSet` should be of length 3, but is of length {len(conlist)}')
    self.assertEqual(len(eq_cons), 2, msg=f'Number of equality constraints for this`AxisAlignedEllipsoidalSet` should be 2, there are {len(eq_cons)} such constraints')