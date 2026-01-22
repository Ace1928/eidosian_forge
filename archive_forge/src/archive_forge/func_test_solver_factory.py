import os
import pyomo.common.unittest as unittest
from pyomo.opt import (
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.opt.plugins.sol import ResultsReader_sol
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC
def test_solver_factory(self):
    """
        Testing the pyomo.opt solver factory with MIP solvers
        """
    SolverFactory.register('stest3')(MockSolver)
    ans = sorted(SolverFactory)
    tmp = ['_mock_asl', '_mock_cbc', '_mock_cplex', '_mock_glpk', 'cbc', 'cplex', 'glpk', 'scip', 'stest3', 'asl']
    tmp.sort()
    self.assertTrue(set(tmp) <= set(ans), msg='Set %s is not a subset of set %s' % (tmp, ans))