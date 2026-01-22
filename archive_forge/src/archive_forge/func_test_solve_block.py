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
@unittest.skipIf(not LOA_solvers_available, 'Required subsolvers %s are not available' % (LOA_solvers,))
def test_solve_block(self):
    m = ConcreteModel()
    m.b = Block()
    m.b.x = Var(bounds=(-5, 5))
    m.b.y = Var(bounds=(-2, 6))
    m.b.disjunction = Disjunction(expr=[[m.b.x + m.b.y <= 1, m.b.y >= 0.5], [m.b.x == 2, m.b.y == 4], [m.b.x ** 2 - m.b.y <= 3]])
    m.disjunction = Disjunction(expr=[[m.b.x - m.b.y <= -2, m.b.y >= -1], [m.b.x == 0, m.b.y >= 0], [m.b.y ** 2 + m.b.x <= 3]])
    m.b.obj = Objective(expr=m.b.x)
    SolverFactory('gdpopt.ric').solve(m.b, mip_solver=mip_solver, nlp_solver=nlp_solver)
    self.assertAlmostEqual(value(m.b.x), -5)
    self.assertEqual(len(m.component_map(Block)), 1)