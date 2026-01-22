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
def eval_parameter_bounds(uncertainty_set, solver):
    """
    Evaluate parameter bounds of uncertainty set by solving
    bounding problems (as opposed to via the `parameter_bounds`
    method).
    """
    bounding_mdl = uncertainty_set.bounding_model()
    param_bounds = []
    for idx, obj in bounding_mdl.param_var_objectives.items():
        obj.activate()
        bounds = []
        for sense in (minimize, maximize):
            obj.sense = sense
            solver.solve(bounding_mdl)
            bounds.append(value(obj))
        param_bounds.append(tuple(bounds))
        obj.sense = minimize
        obj.deactivate()
    return param_bounds