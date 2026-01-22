import json
import os
from os.path import join
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverFactory
from pyomo.core import ConcreteModel, Var, Objective, Constraint
def test_scip_solve_from_instance(self):
    results = self.scip.solve(self.model, suffixes=['.*'])
    self.model.solutions.store_to(results)
    results.Solution(0).Message = 'Scip'
    results.Solver.Message = 'Scip'
    results.Solver.Time = 0
    _out = TempfileManager.create_tempfile('.txt')
    results.write(filename=_out, times=False, format='json')
    self.compare_json(_out, join(currdir, 'test_scip_solve_from_instance.baseline'))