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
def test_identify_mutable_parameters(self):
    m = ConcreteModel()
    m.I = RangeSet(3)
    m.a = Var(initialize=1)
    m.b = Var(m.I, initialize=1)
    self.assertEqual(list(identify_mutable_parameters(m.a)), [])
    self.assertEqual(list(identify_mutable_parameters(m.b[1])), [])
    self.assertEqual(list(identify_mutable_parameters(m.a + m.b[1])), [])
    self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1])), [])
    self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2])), [])
    self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2] * m.b[3] * m.b[2])), [])