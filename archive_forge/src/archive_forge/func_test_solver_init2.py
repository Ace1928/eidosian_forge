import os
import pyomo.common.unittest as unittest
import pyomo.opt
import pyomo.solvers.plugins.solvers
from pyomo.solvers.plugins.solvers.CBCplugin import MockCBC
def test_solver_init2(self):
    """
        Verify that options can be passed in.
        """
    opt = {}
    opt['a'] = 1
    opt['b'] = 'two'
    ans = pyomo.opt.SolverFactory('_mock_cbc', name='solver_init2', options=opt)
    self.assertEqual(ans.options['a'], opt['a'])
    self.assertEqual(ans.options['b'], opt['b'])