import copy
from io import StringIO
from pyomo.core.expr import expr_common
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.core.base.expression import _GeneralExpressionData
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.common.tee import capture_output
def test_init_abstract_nonindexed(self):
    model = AbstractModel()
    model.y = Var(initialize=0.0)
    model.x = Var(initialize=1.0)
    model.ec = Expression(initialize=0.0)

    def obj_rule(model):
        return 1.0 + model.ec
    model.obj = Objective(rule=obj_rule)
    inst = model.create_instance()
    self.assertEqual(inst.obj.expr(), 1.0)
    self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
    e = 1.0
    inst.ec.set_value(e)
    self.assertEqual(inst.obj.expr(), 2.0)
    self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
    e += inst.x
    inst.ec.set_value(e)
    self.assertEqual(inst.obj.expr(), 3.0)
    self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
    e += inst.x
    self.assertEqual(inst.obj.expr(), 3.0)
    self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
    model.del_component('obj')
    model.del_component('ec')
    model.ec = Expression(initialize=0.0)

    def obj_rule(model):
        return 1.0 + model.ec
    model.obj = Objective(rule=obj_rule)
    inst = model.create_instance()
    self.assertEqual(inst.obj.expr(), 1.0)
    self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
    e = 1.0
    inst.ec.set_value(e)
    self.assertEqual(inst.obj.expr(), 2.0)
    self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
    e += inst.x
    inst.ec.set_value(e)
    self.assertEqual(inst.obj.expr(), 3.0)
    self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
    e += inst.x
    self.assertEqual(inst.obj.expr(), 3.0)
    self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
    model.del_component('obj')
    model.del_component('ec')
    model.ec = Expression(initialize=0.0)

    def obj_rule(model):
        return 1.0 + model.ec
    model.obj = Objective(rule=obj_rule)
    inst = model.create_instance()
    self.assertEqual(inst.obj.expr(), 1.0)
    self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
    e = 1.0
    inst.ec.set_value(e)
    self.assertEqual(inst.obj.expr(), 2.0)
    self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
    e += inst.x
    inst.ec.set_value(e)
    self.assertEqual(inst.obj.expr(), 3.0)
    self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))
    e += inst.x
    self.assertEqual(inst.obj.expr(), 3.0)
    self.assertEqual(id(inst.obj.expr.arg(1)), id(inst.ec))