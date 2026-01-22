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
def test_bounds_to_constraints(self):
    m = ConcreteModel()
    m.x = Var(initialize=1, bounds=(0, 1))
    m.y = Var(initialize=0, bounds=(None, 1))
    m.w = Var(initialize=0, bounds=(1, None))
    m.z = Var(initialize=0, bounds=(None, None))
    turn_bounds_to_constraints(m.z, m)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 0, msg='Inequality constraints were written for bounds on a variable with no bounds.')
    turn_bounds_to_constraints(m.y, m)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 1, msg='Inequality constraints were not written correctly for a variable with an upper bound and no lower bound.')
    turn_bounds_to_constraints(m.w, m)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 2, msg='Inequality constraints were not written correctly for a variable with a lower bound and no upper bound.')
    turn_bounds_to_constraints(m.x, m)
    self.assertEqual(len(list(m.component_data_objects(Constraint))), 4, msg='Inequality constraints were not written correctly for a variable with both lower and upper bound.')