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
def test_cloning_negative_case(self):
    """
        Testing correct behavior if incorrect first_stage_vars list object is passed to selective_clone
        """
    m = ConcreteModel()
    m.x = Var(initialize=2)
    m.y = Var(initialize=2)
    m.p = Param(initialize=1)
    m.con = Constraint(expr=m.x * m.p + m.y <= 0)
    n = ConcreteModel()
    n.x = Var()
    m.first_stage_vars = [n.x]
    cloned_model = selective_clone(block=m, first_stage_vars=m.first_stage_vars)
    self.assertNotEqual(id(m.first_stage_vars), id(cloned_model.first_stage_vars), msg='First stage variables should not be equal.')