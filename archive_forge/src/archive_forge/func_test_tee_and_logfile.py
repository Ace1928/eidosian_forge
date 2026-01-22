import pyomo.environ as pyo
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.GAMS import GAMSShell, GAMSDirect, gdxcc_available
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
import os, shutil
from tempfile import mkdtemp
def test_tee_and_logfile(self):
    with SolverFactory('gams', solver_io='python') as opt:
        with capture_output() as output:
            opt.solve(self.m, logfile=self.logfile, tee=True)
    self._check_stdout(output.getvalue(), exists=True)
    self._check_logfile(exists=True)