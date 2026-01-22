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
def test_solve_with_pickle_then_clone(self):
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
    self.assertIn(id(tmodel.b.obj), tmodel.solutions[0]._entry['objective'])
    self.assertIs(tmodel.b.obj, tmodel.solutions[0]._entry['objective'][id(tmodel.b.obj)][0])
    inst = tmodel.clone()
    self.assertTrue(hasattr(inst, 'A'))
    self.assertTrue(hasattr(inst, 'b'))
    self.assertTrue(hasattr(inst.b, 'x'))
    self.assertTrue(hasattr(inst.b, 'obj'))
    self.assertTrue(hasattr(inst, 'c'))
    self.assertIsNot(inst.A, tmodel.A)
    self.assertIsNot(inst.b, tmodel.b)
    self.assertIsNot(inst.b.x, tmodel.b.x)
    self.assertIsNot(inst.b.obj, tmodel.b.obj)
    self.assertIsNot(inst.c, tmodel.c)
    self.assertTrue(hasattr(inst, 'solutions'))
    self.assertEqual(len(inst.solutions), 1)
    self.assertEqual(inst.solutions[0].gap, 0.0)
    self.assertEqual(inst.solutions[0].message, None)
    self.assertIn(id(inst.b.obj), inst.solutions[0]._entry['objective'])
    _obj = inst.solutions[0]._entry['objective'][id(inst.b.obj)]
    self.assertIs(_obj[0], inst.b.obj)
    for v in [1, 2, 3, 4]:
        self.assertIn(id(inst.b.x[v]), inst.solutions[0]._entry['variable'])
        _v = inst.solutions[0]._entry['variable'][id(inst.b.x[v])]
        self.assertIs(_v[0], inst.b.x[v])