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
def test_replacement_linear_expression_with_nonlinear(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    e = LinearExpression(linear_coefs=[2, 3], linear_vars=[m.x, m.y])
    sub_map = dict()
    sub_map[id(m.x)] = m.x ** 2
    e2 = replace_expressions(e, sub_map)
    assertExpressionsEqual(self, e2, SumExpression([2 * m.x ** 2, 3 * m.y]))