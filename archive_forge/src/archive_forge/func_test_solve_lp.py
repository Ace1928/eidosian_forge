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
@unittest.skipUnless(SolverFactory(mip_solver).available(), 'MIP solver not available')
def test_solve_lp(self):
    m = ConcreteModel()
    m.x = Var(bounds=(-5, 5))
    m.c = Constraint(expr=m.x >= 1)
    m.o = Objective(expr=m.x)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
        results = SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver)
        self.assertIn('Your model is an LP (linear program).', output.getvalue().strip())
        self.assertAlmostEqual(value(m.o.expr), 1)
        self.assertEqual(results.problem.number_of_binary_variables, 0)
        self.assertEqual(results.problem.number_of_integer_variables, 0)
        self.assertEqual(results.problem.number_of_disjunctions, 0)
        self.assertAlmostEqual(results.problem.lower_bound, 1)
        self.assertAlmostEqual(results.problem.upper_bound, 1)