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
def test_uncertainty_set_with_incorrect_params(self):
    """
        Case in which the set is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        """
    m = ConcreteModel()
    m.p1 = Var(initialize=0)
    m.p2 = Var(initialize=0)
    m.uncertain_params = [m.p1, m.p2]
    m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
    bounds = [(-1, 1), (-1, 1)]
    Q1 = BoxSet(bounds=bounds)
    Q2 = AxisAlignedEllipsoidalSet(center=[0, 0], half_lengths=[2, 1])
    Q = IntersectionSet(Q1=Q1, Q2=Q2)
    solver = SolverFactory('ipopt')
    config = ConfigBlock()
    config.declare('global_solver', ConfigValue(default=solver))
    m.uncertainty_set_contr = Q.set_as_constraint(uncertain_params=m.uncertain_param_vars, config=config)
    vars_in_expr = []
    for con in m.uncertainty_set_contr.values():
        for v in m.uncertain_param_vars.values():
            if id(v) in [id(u) for u in list(identify_variables(expr=con.expr))]:
                if id(v) not in list((id(u) for u in vars_in_expr)):
                    vars_in_expr.append(v)
    self.assertEqual(len(vars_in_expr), 0, msg='Uncertainty set constraint contains no Var objects, consists of a not potentially variable expression.')