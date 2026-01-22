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
@unittest.skipIf(not yaml_available, 'YAML not available available')
def test_solve_with_store1(self):
    model = ConcreteModel()
    model.A = RangeSet(1, 4)
    model.b = Block()
    model.b.x = Var(model.A, bounds=(-1, 1))
    model.b.obj = Objective(expr=sum_product(model.b.x))
    model.c = Constraint(expr=model.b.x[1] >= 0)
    opt = SolverFactory('glpk')
    results = opt.solve(model, symbolic_solver_labels=True)
    results.write(filename=join(currdir, 'solve_with_store1.out'), format='yaml')
    with open(join(currdir, 'solve_with_store1.out'), 'r') as out, open(join(currdir, 'solve_with_store1.txt'), 'r') as txt:
        self.assertStructuredAlmostEqual(yaml.full_load(txt), yaml.full_load(out), allow_second_superset=True)
    model.solutions.store_to(results)
    results.write(filename=join(currdir, 'solve_with_store2.out'), format='yaml')
    with open(join(currdir, 'solve_with_store2.out'), 'r') as out, open(join(currdir, 'solve_with_store2.txt'), 'r') as txt:
        self.assertStructuredAlmostEqual(yaml.full_load(txt), yaml.full_load(out), allow_second_superset=True)
    tmodel = ConcreteModel()
    tmodel.A = RangeSet(1, 4)
    tmodel.b = Block()
    tmodel.b.x = Var(tmodel.A, bounds=(-1, 1))
    tmodel.b.obj = Objective(expr=sum_product(tmodel.b.x))
    tmodel.c = Constraint(expr=tmodel.b.x[1] >= 0)
    self.assertEqual(len(tmodel.solutions), 0)
    tmodel.solutions.load_from(results)
    self.assertEqual(len(tmodel.solutions), 1)