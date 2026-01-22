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
@unittest.skipUnless(SolverFactory('bonmin').available(exception_flag=False), 'Bonmin is not available')
def test_LBB_8PP_with_screening(self):
    """Test the logic-based branch and bound algorithm."""
    exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
    eight_process = exfile.build_eight_process_flowsheet()
    results = SolverFactory('gdpopt.lbb').solve(eight_process, tee=False, minlp_solver=minlp_solver, minlp_solver_args=minlp_args, solve_local_rnGDP=True, local_minlp_solver='bonmin', local_minlp_solver_args={})
    ct.check_8PP_solution(self, eight_process, results)