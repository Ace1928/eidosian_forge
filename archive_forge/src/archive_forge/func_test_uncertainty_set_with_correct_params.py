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
@unittest.skipUnless(ipopt_available, 'IPOPT is not available.')
def test_uncertainty_set_with_correct_params(self):
    """
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        """
    m = ConcreteModel()
    m.p1 = Var(initialize=0)
    m.p2 = Var(initialize=0)
    m.uncertain_params = [m.p1, m.p2]
    m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
    bounds = [(-1, 1), (-1, 1)]
    Q1 = BoxSet(bounds=bounds)
    Q2 = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
    Q = IntersectionSet(Q1=Q1, Q2=Q2)
    config = ConfigBlock()
    solver = SolverFactory('ipopt')
    config.declare('global_solver', ConfigValue(default=solver))
    m.uncertainty_set_contr = Q.set_as_constraint(uncertain_params=m.uncertain_param_vars, config=config)
    uncertain_params_in_expr = []
    for con in m.uncertainty_set_contr.values():
        for v in m.uncertain_param_vars.values():
            if v in ComponentSet(identify_variables(expr=con.expr)):
                if id(v) not in list((id(u) for u in uncertain_params_in_expr)):
                    uncertain_params_in_expr.append(v)
    self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()], msg='Uncertain param Var objects used to construct uncertainty set constraint must be the same uncertain param Var objects in the original model.')