from io import StringIO
import os
import sys
import types
import json
from copy import deepcopy
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.block import (
import pyomo.core.expr as EXPR
from pyomo.opt import check_available_solvers
from pyomo.gdp import Disjunct
@unittest.skipIf(not 'glpk' in solvers, 'glpk solver is not available')
def test_solve1(self):
    model = Block(concrete=True)
    model.A = RangeSet(1, 4)
    model.x = Var(model.A, bounds=(-1, 1))

    def obj_rule(model):
        return sum_product(model.x)
    model.obj = Objective(rule=obj_rule)

    def c_rule(model):
        expr = 0
        for i in model.A:
            expr += i * model.x[i]
        return expr == 0
    model.c = Constraint(rule=c_rule)
    opt = SolverFactory('glpk')
    results = opt.solve(model, symbolic_solver_labels=True)
    model.solutions.store_to(results)
    results.write(filename=join(currdir, 'solve1.out'), format='json')
    with open(join(currdir, 'solve1.out'), 'r') as out, open(join(currdir, 'solve1.txt'), 'r') as txt:
        self.assertStructuredAlmostEqual(json.load(txt), json.load(out), abstol=0.0001, allow_second_superset=True)

    def d_rule(model):
        return model.x[1] >= 0
    model.d = Constraint(rule=d_rule)
    model.d.deactivate()
    results = opt.solve(model)
    model.solutions.store_to(results)
    results.write(filename=join(currdir, 'solve1x.out'), format='json')
    with open(join(currdir, 'solve1x.out'), 'r') as out, open(join(currdir, 'solve1.txt'), 'r') as txt:
        self.assertStructuredAlmostEqual(json.load(txt), json.load(out), abstol=0.0001, allow_second_superset=True)
    model.d.activate()
    results = opt.solve(model)
    model.solutions.store_to(results)
    results.write(filename=join(currdir, 'solve1a.out'), format='json')
    with open(join(currdir, 'solve1a.out'), 'r') as out, open(join(currdir, 'solve1a.txt'), 'r') as txt:
        self.assertStructuredAlmostEqual(json.load(txt), json.load(out), abstol=0.0001, allow_second_superset=True)
    model.d.deactivate()

    def e_rule(model, i):
        return model.x[i] >= 0
    model.e = Constraint(model.A, rule=e_rule)
    for i in model.A:
        model.e[i].deactivate()
    results = opt.solve(model)
    model.solutions.store_to(results)
    results.write(filename=join(currdir, 'solve1y.out'), format='json')
    with open(join(currdir, 'solve1y.out'), 'r') as out, open(join(currdir, 'solve1.txt'), 'r') as txt:
        self.assertStructuredAlmostEqual(json.load(txt), json.load(out), abstol=0.0001, allow_second_superset=True)
    model.e.activate()
    results = opt.solve(model)
    model.solutions.store_to(results)
    results.write(filename=join(currdir, 'solve1b.out'), format='json')
    with open(join(currdir, 'solve1b.out'), 'r') as out, open(join(currdir, 'solve1b.txt'), 'r') as txt:
        self.assertStructuredAlmostEqual(json.load(txt), json.load(out), abstol=0.0001, allow_second_superset=True)