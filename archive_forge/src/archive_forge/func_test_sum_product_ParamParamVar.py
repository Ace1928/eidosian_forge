import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.compare import assertExpressionsEqual
def test_sum_product_ParamParamVar(self):
    model = AbstractModel()
    model.A = Set(initialize=[1, 2, 3])
    model.B = Param(model.A, initialize={1: 100, 2: 200, 3: 300}, mutable=True)
    model.x = Var(model.A)
    model.y = Param(model.A, mutable=True)
    instance = model.create_instance()
    expr = sum_product(instance.B, instance.y, instance.x)
    baseline = 'B[1]*y[1]*x[1] + B[2]*y[2]*x[2] + B[3]*y[3]*x[3]'
    self.assertEqual(str(expr), baseline)