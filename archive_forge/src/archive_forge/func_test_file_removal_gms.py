import pyomo.environ as pyo
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.GAMS import GAMSShell, GAMSDirect, gdxcc_available
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.tee import capture_output
import os, shutil
from tempfile import mkdtemp
@unittest.skipIf(not gamsgms_available, "The 'gams' executable is not available")
def test_file_removal_gms(self):
    with SolverFactory('gams', solver_io='gms') as opt:
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=m.x >= 10)
        m.o = Objective(expr=m.x)
        tmpdir = mkdtemp()
        results = opt.solve(m, tmpdir=tmpdir)
        self.assertTrue(os.path.exists(tmpdir))
        self.assertFalse(os.path.exists(os.path.join(tmpdir, 'model.gms')))
        self.assertFalse(os.path.exists(os.path.join(tmpdir, 'output.lst')))
        self.assertFalse(os.path.exists(os.path.join(tmpdir, 'GAMS_MODEL_p.gdx')))
        self.assertFalse(os.path.exists(os.path.join(tmpdir, 'GAMS_MODEL_s.gdx')))
        os.rmdir(tmpdir)
        results = opt.solve(m, tmpdir=tmpdir)
        self.assertFalse(os.path.exists(tmpdir))