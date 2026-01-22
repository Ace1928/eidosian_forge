from pyomo.common.dependencies import (
import platform
import pyomo.common.unittest as unittest
import os
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest as parmestbase
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.common.fileutils import find_library
def test_ipopt_solve_with_stats(self):
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import rooney_biegler_model
    from pyomo.contrib.parmest.utils import ipopt_solve_with_stats
    data = pd.DataFrame(data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]], columns=['hour', 'y'])
    model = rooney_biegler_model(data)
    solver = pyo.SolverFactory('ipopt')
    solver.solve(model)
    status_obj, solved, iters, time, regu = ipopt_solve_with_stats(model, solver)
    self.assertEqual(solved, True)