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
def test_nestedProduct3(self):
    m = AbstractModel()
    m.a = Param(mutable=True)
    m.b = Var()
    m.c = Var()
    m.d = Var()
    e1 = 3 * m.b
    e = e1 * 5
    self.assertIs(type(e), MonomialTermExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertEqual(e.arg(0), 15)
    self.assertIs(e.arg(1), m.b)
    self.assertEqual(e.size(), 3)
    e1 = m.a * m.b
    e = e1 * 5
    self.assertExpressionsEqual(e, MonomialTermExpression((NPV_ProductExpression((m.a, 5)), m.b)))
    e1 = 3 * m.b
    e = 5 * e1
    self.assertIs(type(e), MonomialTermExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertEqual(e.arg(0), 15)
    self.assertIs(e.arg(1), m.b)
    self.assertEqual(e.size(), 3)
    e1 = m.a * m.b
    e = 5 * e1
    self.assertIs(type(e), MonomialTermExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(type(e.arg(0)), NPV_ProductExpression)
    self.assertEqual(e.arg(0).arg(0), 5)
    self.assertIs(e.arg(0).arg(1), m.a)
    self.assertIs(e.arg(1), m.b)
    self.assertEqual(e.size(), 5)
    e1 = m.a * m.b
    e = e1 * m.c
    self.assertIs(type(e), ProductExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(e.arg(1), m.c)
    self.assertIs(type(e.arg(0)), MonomialTermExpression)
    self.assertIs(e.arg(0).arg(0), m.a)
    self.assertIs(e.arg(0).arg(1), m.b)
    self.assertEqual(e.size(), 5)
    e1 = m.a * m.b
    e = m.c * e1
    self.assertIs(type(e), ProductExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(e.arg(0), m.c)
    self.assertIs(type(e.arg(1)), MonomialTermExpression)
    self.assertIs(e.arg(1).arg(0), m.a)
    self.assertIs(e.arg(1).arg(1), m.b)
    self.assertEqual(e.size(), 5)
    e1 = m.a * m.b
    e2 = m.c * m.d
    e = e1 * e2
    self.assertIs(type(e), ProductExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(type(e.arg(0)), MonomialTermExpression)
    self.assertIs(type(e.arg(1)), ProductExpression)
    self.assertIs(e.arg(0).arg(0), m.a)
    self.assertIs(e.arg(0).arg(1), m.b)
    self.assertIs(e.arg(1).arg(0), m.c)
    self.assertIs(e.arg(1).arg(1), m.d)
    self.assertEqual(e.size(), 7)