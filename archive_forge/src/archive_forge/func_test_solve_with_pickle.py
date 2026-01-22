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
@unittest.skipIf('glpk' not in solvers, 'glpk solver is not available')
def test_solve_with_pickle(self):
    model = ConcreteModel()
    model.A = RangeSet(1, 4)
    model.b = Block()
    model.b.x = Var(model.A, bounds=(-1, 1))
    model.b.obj = Objective(expr=sum_product(model.b.x))
    model.c = Constraint(expr=model.b.x[1] >= 0)
    opt = SolverFactory('glpk')
    self.assertEqual(len(model.solutions), 0)
    results = opt.solve(model, symbolic_solver_labels=True)
    self.assertEqual(len(model.solutions), 1)
    self.assertEqual(model.solutions[0].gap, 0.0)
    self.assertEqual(model.solutions[0].message, None)
    buf = pickle.dumps(model)
    tmodel = pickle.loads(buf)
    self.assertEqual(len(tmodel.solutions), 1)
    self.assertEqual(tmodel.solutions[0].gap, 0.0)
    self.assertEqual(tmodel.solutions[0].message, None)