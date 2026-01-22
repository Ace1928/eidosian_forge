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
def test_bound_infeasibility_in_subproblems(self):
    m = ConcreteModel()
    m.x = Var(bounds=(2, 4))
    m.y = Var(bounds=(5, 10))
    m.disj = Disjunction(expr=[[m.x == m.y, m.x + m.y >= 8], [m.x == 4]])
    m.obj = Objective(expr=m.x + m.y)
    SolverFactory('gdpopt.ric').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver, init_algorithm='set_covering', tee=True)
    self.assertAlmostEqual(value(m.x), 4)
    self.assertAlmostEqual(value(m.y), 5)
    self.assertFalse(value(m.disj.disjuncts[0].indicator_var))
    self.assertTrue(value(m.disj.disjuncts[1].indicator_var))