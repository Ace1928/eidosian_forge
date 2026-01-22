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
def test_summation4(self):
    e = quicksum((self.m.a[i] * self.m.b[i] for i in self.m.a))
    self.assertEqual(e(), 250)
    self.assertExpressionsEqual(e, SumExpression([ProductExpression((self.m.a[1], self.m.b[1])), ProductExpression((self.m.a[2], self.m.b[2])), ProductExpression((self.m.a[3], self.m.b[3])), ProductExpression((self.m.a[4], self.m.b[4])), ProductExpression((self.m.a[5], self.m.b[5]))]))