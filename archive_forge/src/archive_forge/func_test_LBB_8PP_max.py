from io import StringIO
import logging
from math import fabs
from os.path import abspath, dirname, join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import import_file
from pyomo.common.log import LoggingIntercept
import pyomo.contrib.gdpopt.tests.common_tests as ct
from pyomo.contrib.satsolver.satsolver import z3_available
from pyomo.environ import SolverFactory, value, ConcreteModel, Var, Objective, maximize
from pyomo.gdp import Disjunction
from pyomo.opt import TerminationCondition
@unittest.skipUnless(license_available, 'Problem is too big for unlicensed BARON.')
def test_LBB_8PP_max(self):
    """Test the logic-based branch and bound algorithm."""
    exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
    eight_process = exfile.build_eight_process_flowsheet()
    obj = next(eight_process.component_data_objects(Objective, active=True))
    obj.sense = maximize
    obj.set_value(-1 * obj.expr)
    SolverFactory('gdpopt.lbb').solve(eight_process, tee=False, minlp_solver=minlp_solver, minlp_solver_args=minlp_args)
    self.assertAlmostEqual(value(eight_process.profit.expr), -68, places=1)