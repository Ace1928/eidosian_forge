import json
import os
from os.path import join
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverFactory
from pyomo.core import ConcreteModel, Var, Objective, Constraint
def test_scip_solve_from_instance_options(self):
    _cwd = os.getcwd()
    tmpdir = TempfileManager.create_tempdir()
    try:
        os.chdir(tmpdir)
        open(join(tmpdir, 'scip.set'), 'w').close()
        with LoggingIntercept() as LOG:
            results = self.scip.solve(self.model, suffixes=['.*'], options={'limits/softtime': 100})
        self.assertRegex(LOG.getvalue().replace('\n', ' '), 'A file named (.*) exists in the current working directory, but SCIP options are being set using a separate options file. The options file \\1 will be ignored.')
    finally:
        os.chdir(_cwd)
    self.model.solutions.store_to(results)
    results.Solution(0).Message = 'Scip'
    results.Solver.Message = 'Scip'
    results.Solver.Time = 0
    _out = TempfileManager.create_tempfile('.txt')
    results.write(filename=_out, times=False, format='json')
    self.compare_json(_out, join(currdir, 'test_scip_solve_from_instance.baseline'))