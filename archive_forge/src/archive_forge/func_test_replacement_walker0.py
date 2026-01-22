import os
import platform
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import native_types, nonpyomo_leaf_types, NumericConstant
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import (
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.core.expr.compare import assertExpressionsEqual
def test_replacement_walker0(self):
    M = ConcreteModel()
    M.x = Var(range(3))
    M.w = VarList()
    M.z = Param(range(3), mutable=True)
    e = sum_product(M.z, M.x)
    self.assertIs(type(e), LinearExpression)
    walker = ReplacementWalkerTest3(M)
    f = walker.dfs_postorder_stack(e)
    assertExpressionsEqual(self, LinearExpression(linear_coefs=[i for i in M.z.values()], linear_vars=[i for i in M.x.values()]), e)
    assertExpressionsEqual(self, 2 * M.w[1] * M.x[0] + 2 * M.w[2] * M.x[1] + 2 * M.w[3] * M.x[2], f)
    e = 2 * sum_product(M.z, M.x)
    walker = ReplacementWalkerTest3(M)
    f = walker.dfs_postorder_stack(e)
    assertExpressionsEqual(self, 2 * LinearExpression(linear_coefs=[i for i in M.z.values()], linear_vars=[i for i in M.x.values()]), e)
    assertExpressionsEqual(self, 2 * (2 * M.w[4] * M.x[0] + 2 * M.w[5] * M.x[1] + 2 * M.w[6] * M.x[2]), f)