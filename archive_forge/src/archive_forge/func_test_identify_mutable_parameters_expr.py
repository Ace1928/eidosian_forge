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
def test_identify_mutable_parameters_expr(self):
    m = ConcreteModel()
    m.a = Param(initialize=1, mutable=True)
    m.b = Param(initialize=2, mutable=True)
    m.e = Expression(expr=3 * m.a)
    m.E = Expression([0, 1], initialize={0: 3 * m.a, 1: 4 * m.b})
    self.assertEqual(list(identify_mutable_parameters(m.b + m.e)), [m.b, m.a])
    self.assertEqual(list(identify_mutable_parameters(m.E[0])), [m.a])
    self.assertEqual(list(identify_mutable_parameters(m.E[1])), [m.b])