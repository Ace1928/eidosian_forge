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
def test_exitNode(self):

    def exit(node, data):
        if data:
            return data
        else:
            return [node]
    walker = StreamBasedExpressionVisitor(exitNode=exit)
    m = self.m
    ans = self.walk(walker, self.e)
    ref = [[[m.x], [2]], [m.y], [[m.z], [[m.x], [m.y]]]]
    self.assertEqual(str(ans), str(ref))
    ans = self.walk(walker, m.x)
    ref = [m.x]
    self.assertEqual(str(ans), str(ref))
    ans = self.walk(walker, 2)
    ref = [2]
    self.assertEqual(str(ans), str(ref))