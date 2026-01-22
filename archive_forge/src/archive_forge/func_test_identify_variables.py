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
def test_identify_variables(self):
    M = ConcreteModel()
    M.x = Var()
    M.y = Var()
    M.w = Var()
    M.w = 2
    M.w.fixed = True
    e = sin(M.x) + M.x * M.w + 3
    v = list((str(v) for v in identify_variables(e)))
    self.assertEqual(v, ['x', 'w'])
    v = list((str(v) for v in identify_variables(e, include_fixed=False)))
    self.assertEqual(v, ['x'])