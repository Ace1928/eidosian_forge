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
def test_transform_does_not_alter_num_of_constraints(self):
    """
        Check that if model does not contain any constraints
        for which both the `lower` and `upper` attributes are
        distinct and not None, then number of constraints remains the same
        after constraint standardization.
        Standard form for the purpose of PyROS is all inequality constraints
        as `g(.)<=0`.
        """
    m = ConcreteModel()
    m.x = Var(initialize=1, bounds=(0, 1))
    m.y = Var(initialize=0, bounds=(None, 1))
    m.con1 = Constraint(expr=m.x >= 1 + m.y)
    m.con2 = Constraint(expr=m.x ** 2 + m.y ** 2 >= 9)
    original_num_constraints = len(list(m.component_data_objects(Constraint)))
    transform_to_standard_form(m)
    final_num_constraints = len(list(m.component_data_objects(Constraint)))
    self.assertEqual(original_num_constraints, final_num_constraints, msg='Transform to standard form function led to a different number of constraints than in the original model.')
    number_of_non_standard_form_inequalities = len(list((c for c in list(m.component_data_objects(Constraint)) if c.lower != None)))
    self.assertEqual(number_of_non_standard_form_inequalities, 0, msg='All inequality constraints were not transformed to standard form.')