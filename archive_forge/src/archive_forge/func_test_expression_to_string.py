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
def test_expression_to_string(self):
    M = ConcreteModel()
    M.x = Var()
    M.w = Var()
    e = sin(M.x) + M.x * M.w + 3
    self.assertEqual('sin(x) + x*w + 3', expression_to_string(e))
    M.w = 2
    M.w.fixed = True
    self.assertEqual('sin(x) + x*2 + 3', expression_to_string(e, compute_values=True))