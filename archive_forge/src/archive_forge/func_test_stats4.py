import json
import os
from os.path import abspath, dirname, join
import pickle
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml_available
from pyomo.common.tempfiles import TempfileManager
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.opt import check_available_solvers
from pyomo.opt.parallel.local import SolverManager_Serial
def test_stats4(self):
    model = ConcreteModel()
    model.x = Var([1])
    model.B = Block()
    model.B.x = Var([1, 2, 3])
    model.B.o = ObjectiveList()
    model.B.o.add(model.B.x[1])
    model.B.c = ConstraintList()
    model.B.c.add(model.B.x[1] == 0)
    model.B.c.add(model.B.x[2] == 0)
    model.B.c.add(model.B.x[3] == 0)
    self.assertEqual(model.nvariables(), 4)
    self.assertEqual(model.nobjectives(), 1)
    self.assertEqual(model.nconstraints(), 3)
    model.clear()
    self.assertEqual(model.nvariables(), 0)
    self.assertEqual(model.nobjectives(), 0)
    self.assertEqual(model.nconstraints(), 0)