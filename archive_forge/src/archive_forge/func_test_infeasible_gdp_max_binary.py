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
def test_infeasible_gdp_max_binary(self):
    """Test that max binary initialization catches infeasible GDP too"""
    m = models.make_infeasible_gdp_model()
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.DEBUG):
        results = SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver, init_algorithm='max_binary')
        self.assertIn('MILP relaxation for initialization was infeasible. Problem is infeasible.', output.getvalue().strip())
    self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasible)