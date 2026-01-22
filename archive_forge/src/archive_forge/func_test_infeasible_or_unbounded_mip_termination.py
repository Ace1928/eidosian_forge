from contextlib import redirect_stdout
from io import StringIO
import logging
from math import fabs
from os.path import join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import Bunch
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.contrib.appsi.solvers.gurobi import Gurobi
from pyomo.contrib.gdpopt.create_oa_subproblems import (
import pyomo.contrib.gdpopt.tests.common_tests as ct
from pyomo.contrib.gdpopt.util import is_feasible, time_code
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.tests import models
from pyomo.opt import TerminationCondition
@unittest.skipUnless(gurobi_available, 'Gurobi not available')
def test_infeasible_or_unbounded_mip_termination(self):
    m = ConcreteModel()
    m.x = Var()
    m.c1 = Constraint(expr=m.x >= 2)
    m.c2 = Constraint(expr=m.x <= 1.9)
    m.obj = Objective(expr=m.x)
    results = SolverFactory('gurobi').solve(m)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasibleOrUnbounded)
    config = ConfigDict()
    config.declare('mip_solver', ConfigValue('gurobi'))
    config.declare('mip_solver_args', ConfigValue({}))
    results, termination_condition = distinguish_mip_infeasible_or_unbounded(m, config)
    self.assertEqual(termination_condition, TerminationCondition.infeasible)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasible)