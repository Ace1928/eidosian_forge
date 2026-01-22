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
def test_replace_expressions_with_monomial_term(self):
    M = ConcreteModel()
    M.x = Var()
    e = 2.0 * M.x
    substitution_map = {id(M.x): 3.0 * M.x}
    new_e = replace_expressions(e, substitution_map=substitution_map)
    self.assertEqual('6.0*x', str(new_e))