import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.compare import assertExpressionsEqual
def test_expr5(self):
    model = ConcreteModel()
    model.A = Set(initialize=[1, 2, 3], doc='set A')
    model.B = Param(model.A, initialize={1: 100, 2: 200, 3: 300}, doc='param B', mutable=True)
    model.C = Param(initialize=3, doc='param C', mutable=True)
    model.x = Var(model.A, doc='var x')
    model.y = Var(doc='var y')
    model.o = Objective(expr=model.y, doc='obj o')
    model.c1 = Constraint(expr=model.x[1] >= 0, doc='con c1')

    def c2_rule(model, a):
        return model.B[a] * model.x[a] <= 1
    model.c2 = Constraint(model.A, doc='con c2', rule=c2_rule)
    model.c3 = ConstraintList(doc='con c3')
    model.c3.add(model.y <= 0)
    OUTPUT = open(join(currdir, 'test_expr5.out'), 'w')
    model.pprint(ostream=OUTPUT)
    OUTPUT.close()
    _out, _txt = (join(currdir, 'test_expr5.out'), join(currdir, 'test_expr5.txt'))
    self.assertTrue(cmp(_out, _txt), msg='Files %s and %s differ' % (_out, _txt))