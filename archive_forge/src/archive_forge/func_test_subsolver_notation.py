import pyomo.environ as pyo
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.GAMS import GAMSShell, GAMSDirect, gdxcc_available
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
import os, shutil
from tempfile import mkdtemp
def test_subsolver_notation(self):
    opt1 = SolverFactory('gams:ipopt', solver_io='gms')
    self.assertTrue(isinstance(opt1, GAMSShell))
    self.assertEqual(opt1.options['solver'], 'ipopt')
    opt2 = SolverFactory('gams:baron', solver_io='python')
    self.assertTrue(isinstance(opt2, GAMSDirect))
    self.assertEqual(opt2.options['solver'], 'baron')
    opt3 = SolverFactory('py:gams')
    self.assertTrue(isinstance(opt3, GAMSDirect))
    opt3.options['keepfiles'] = True
    self.assertEqual(opt3.options['keepfiles'], True)
    opt4 = SolverFactory('py:gams:cbc')
    self.assertTrue(isinstance(opt4, GAMSDirect))
    self.assertEqual(opt4.options['solver'], 'cbc')