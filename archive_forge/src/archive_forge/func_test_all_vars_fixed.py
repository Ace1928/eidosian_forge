import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.environ as pe
from pyomo.contrib import appsi
from pyomo.contrib.appsi.cmodel import cmodel_available
import os
def test_all_vars_fixed(self):
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    m.obj = pe.Objective(expr=m.x ** 2 + m.y ** 2)
    m.c1 = pe.Constraint(expr=m.y >= pe.exp(m.x))
    m.c2 = pe.Constraint(expr=m.y >= (m.x - 1) ** 2)
    m.x.fix(1)
    m.y.fix(2)
    writer = appsi.writers.NLWriter()
    with TempfileManager:
        fname = TempfileManager.create_tempfile(suffix='.appsi.nl')
        with self.assertRaisesRegex(ValueError, 'there are not any unfixed variables in the problem'):
            writer.write(m, fname)