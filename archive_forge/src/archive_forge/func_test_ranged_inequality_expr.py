import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
def test_ranged_inequality_expr(self):
    model = ConcreteModel()
    model.v = Var()
    model.l = Param(initialize=1, mutable=True)
    model.u = Param(initialize=3, mutable=True)
    model.con = Constraint(expr=inequality(model.l, model.v, model.u))
    self.assertIs(model.con.expr.args[0], model.l)
    self.assertIs(model.con.expr.args[1], model.v)
    self.assertIs(model.con.expr.args[2], model.u)