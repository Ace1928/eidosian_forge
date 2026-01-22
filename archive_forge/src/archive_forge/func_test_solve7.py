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
def test_solve7(self):
    model = Block(concrete=True)
    model.y = Var(bounds=(-1, 1))
    model.A = RangeSet(1, 4)
    model.B = Set(initialize=['A B', 'C,D', 'E'])
    model.x = Var(model.A, model.B, bounds=(-1, 1))

    def obj_rule(model):
        return sum_product(model.x)
    model.obj = Objective(rule=obj_rule)

    def c_rule(model):
        expr = model.y
        for i in model.A:
            for j in model.B:
                expr += i * model.x[i, j]
        return expr == 0
    model.c = Constraint(rule=c_rule)
    opt = SolverFactory('glpk')
    results = opt.solve(model, symbolic_solver_labels=True)
    model.solutions.store_to(results)
    results.write(filename=join(currdir, 'solve7.out'), format='json')
    with open(join(currdir, 'solve7.out'), 'r') as out, open(join(currdir, 'solve7.txt'), 'r') as txt:
        self.assertStructuredAlmostEqual(json.load(txt), json.load(out), abstol=0.0001, allow_second_superset=True)