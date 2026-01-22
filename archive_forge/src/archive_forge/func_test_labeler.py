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
def test_labeler(self):
    M = ConcreteModel()
    M.x = Var()
    M.y = Var()
    M.z = Var()
    M.a = Var(range(3))
    M.p = Param(range(3), initialize=2)
    M.q = Param(range(3), initialize=3, mutable=True)
    e = M.x * M.y + sum_product(M.p, M.a) + quicksum((M.q[i] * M.a[i] for i in M.a)) / M.x
    self.assertEqual(str(e), 'x*y + (2*a[0] + 2*a[1] + 2*a[2]) + (q[0]*a[0] + q[1]*a[1] + q[2]*a[2])/x')
    self.assertEqual(e.to_string(), 'x*y + (2*a[0] + 2*a[1] + 2*a[2]) + (q[0]*a[0] + q[1]*a[1] + q[2]*a[2])/x')
    self.assertEqual(e.to_string(compute_values=True), 'x*y + (2*a[0] + 2*a[1] + 2*a[2]) + (3*a[0] + 3*a[1] + 3*a[2])/x')
    labeler = NumericLabeler('x')
    self.assertEqual(expression_to_string(e, labeler=labeler), 'x1*x2 + (2*x3 + 2*x4 + 2*x5) + (x6*x3 + x7*x4 + x8*x5)/x1')
    from pyomo.core.expr.symbol_map import SymbolMap
    labeler = NumericLabeler('x')
    smap = SymbolMap(labeler)
    self.assertEqual(expression_to_string(e, smap=smap), 'x1*x2 + (2*x3 + 2*x4 + 2*x5) + (x6*x3 + x7*x4 + x8*x5)/x1')
    self.assertEqual(expression_to_string(e, smap=smap, compute_values=True), 'x1*x2 + (2*x3 + 2*x4 + 2*x5) + (3*x3 + 3*x4 + 3*x5)/x1')