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
def test_simple_product(self):
    expr = self.instance.c * self.instance.d
    self.assertEqual(expr.is_fixed(), True)
    self.assertEqual(expr.is_constant(), False)
    self.assertEqual(expr.is_potentially_variable(), False)
    expr = self.instance.a * self.instance.c
    self.assertEqual(expr.is_fixed(), False)
    self.assertEqual(expr.is_constant(), False)
    self.assertEqual(expr.is_potentially_variable(), True)
    expr = self.instance.f * self.instance.b
    self.assertEqual(expr.is_fixed(), True)
    self.assertEqual(expr.is_constant(), False)
    self.assertEqual(expr.is_potentially_variable(), True)
    expr = self.instance.a * self.instance.b
    self.assertEqual(expr.is_fixed(), False)
    self.assertEqual(expr.is_constant(), False)
    self.assertEqual(expr.is_potentially_variable(), True)
    self.instance.a.fixed = True
    self.assertEqual(expr.is_fixed(), False)
    self.assertEqual(expr.is_constant(), False)
    self.assertEqual(expr.is_potentially_variable(), True)
    self.instance.b.fixed = True
    self.assertEqual(expr.is_fixed(), True)
    self.assertEqual(expr.is_constant(), False)
    self.instance.a.fixed = False
    self.assertEqual(expr.is_potentially_variable(), True)
    expr = self.instance.a * self.instance.g
    self.instance.a.fixed = False
    self.instance.g.fixed = True
    self.assertEqual(expr.is_fixed(), True)
    self.assertEqual(expr.is_constant(), False)
    self.assertEqual(expr.is_potentially_variable(), True)
    expr = self.instance.a / self.instance.c
    self.assertEqual(expr.is_fixed(), False)
    self.assertEqual(expr.is_constant(), False)
    self.assertEqual(expr.is_potentially_variable(), True)
    self.instance.a.fixed = True
    self.assertEqual(expr.is_fixed(), True)
    self.assertEqual(expr.is_constant(), False)
    self.assertEqual(expr.is_potentially_variable(), True)
    self.instance.a.fixed = False
    expr = self.instance.c / self.instance.a
    self.assertEqual(expr.is_fixed(), False)
    self.assertEqual(expr.is_constant(), False)
    self.assertEqual(expr.is_potentially_variable(), True)
    self.instance.a.fixed = True
    self.assertEqual(expr.is_fixed(), True)
    self.assertEqual(expr.is_constant(), False)
    self.assertEqual(expr.is_potentially_variable(), True)