import copy
import pickle
import math
import os
from collections import defaultdict
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.environ import (
from pyomo.kernel import variable, expression, objective
from pyomo.core.expr.expr_common import ExpressionType, clone_counter
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import RelationalExpression, EqualityExpression
from pyomo.core.expr.relational_expr import RelationalExpression, EqualityExpression
from pyomo.common.errors import PyomoException
from pyomo.core.expr.visitor import expression_to_string, clone_expression
from pyomo.core.expr import Expr_if
from pyomo.core.base.label import NumericLabeler
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr import expr_common
from pyomo.core.base.var import _GeneralVarData
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numvalue import NumericValue
def test_large_expression(self):

    def c1_rule(model):
        return (1.0, model.b[1], None)

    def c2_rule(model):
        return (None, model.b[1], 0.0)

    def c3_rule(model):
        return (0.0, model.b[1], 1.0)

    def c4_rule(model):
        return (3.0, model.b[1])

    def c5_rule(model, i):
        return (model.b[i], 0.0)

    def c6a_rule(model):
        return 0.0 <= model.c

    def c7a_rule(model):
        return model.c <= 1.0

    def c7b_rule(model):
        return model.c >= 1.0

    def c8_rule(model):
        return model.c == 2.0

    def c9a_rule(model):
        return model.A + model.A <= model.c

    def c9b_rule(model):
        return model.A + model.A >= model.c

    def c10a_rule(model):
        return model.c <= model.B + model.B

    def c11_rule(model):
        return model.c == model.A + model.B

    def c15a_rule(model):
        return model.A <= model.A * model.d

    def c16a_rule(model):
        return model.A * model.d <= model.B

    def c12_rule(model):
        return model.c == model.d

    def c13a_rule(model):
        return model.c <= model.d

    def c14a_rule(model):
        return model.c >= model.d

    def cl_rule(model, i):
        if i > 10:
            return ConstraintList.End
        return i * model.c >= model.d

    def o2_rule(model, i):
        return model.b[i]
    model = AbstractModel()
    model.a = Set(initialize=[1, 2, 3])
    model.b = Var(model.a, initialize=1.1, within=PositiveReals)
    model.c = Var(initialize=2.1, within=PositiveReals)
    model.d = Var(initialize=3.1, within=PositiveReals)
    model.e = Var(initialize=4.1, within=PositiveReals)
    model.A = Param(default=-1, mutable=True)
    model.B = Param(default=-2, mutable=True)
    model.o2 = Objective(model.a, rule=o2_rule)
    model.o3 = Objective(model.a, model.a)
    model.c1 = Constraint(rule=c1_rule)
    model.c2 = Constraint(rule=c2_rule)
    model.c3 = Constraint(rule=c3_rule)
    model.c4 = Constraint(rule=c4_rule)
    model.c5 = Constraint(model.a, rule=c5_rule)
    model.c6a = Constraint(rule=c6a_rule)
    model.c7a = Constraint(rule=c7a_rule)
    model.c7b = Constraint(rule=c7b_rule)
    model.c8 = Constraint(rule=c8_rule)
    model.c9a = Constraint(rule=c9a_rule)
    model.c9b = Constraint(rule=c9b_rule)
    model.c10a = Constraint(rule=c10a_rule)
    model.c11 = Constraint(rule=c11_rule)
    model.c15a = Constraint(rule=c15a_rule)
    model.c16a = Constraint(rule=c16a_rule)
    model.c12 = Constraint(rule=c12_rule)
    model.c13a = Constraint(rule=c13a_rule)
    model.c14a = Constraint(rule=c14a_rule)
    model.cl = ConstraintList(rule=cl_rule)
    instance = model.create_instance()
    OUTPUT = open(join(currdir, 'varpprint.out'), 'w')
    instance.pprint(ostream=OUTPUT)
    OUTPUT.close()
    _out, _txt = (join(currdir, 'varpprint.out'), join(currdir, 'varpprint.txt'))
    self.assertTrue(cmp(_out, _txt), msg='Files %s and %s differ' % (_txt, _out))