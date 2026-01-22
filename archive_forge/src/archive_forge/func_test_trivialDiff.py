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
def test_trivialDiff(self):
    m = ConcreteModel()
    m.a = Var()
    m.p = Param(mutable=True)
    e = m.a - 0
    self.assertIs(type(e), type(m.a))
    self.assertIs(e, m.a)
    e = 0 - m.a
    self.assertIs(type(e), MonomialTermExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertEqual(e.arg(0), -1)
    self.assertIs(e.arg(1), m.a)
    e = m.p - 0
    self.assertIs(type(e), type(m.p))
    self.assertIs(e, m.p)
    e = 0 - m.p
    self.assertIs(type(e), NPV_NegationExpression)
    self.assertEqual(e.nargs(), 1)
    self.assertIs(e.arg(0), m.p)
    e = 0 - 5 * m.a
    self.assertIs(type(e), MonomialTermExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertEqual(e.arg(0), -5)
    e = 0 - m.p * m.a
    self.assertIs(type(e), MonomialTermExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(type(e.arg(0)), NPV_NegationExpression)
    self.assertIs(e.arg(0).arg(0), m.p)
    e = 0 - m.a * m.a
    self.assertIs(type(e), NegationExpression)
    self.assertEqual(e.nargs(), 1)
    self.assertIs(type(e.arg(0)), ProductExpression)