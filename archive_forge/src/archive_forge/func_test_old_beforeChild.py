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
def test_old_beforeChild(self):

    def before(node, child):
        if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
            return (False, [child])
    os = StringIO()
    with LoggingIntercept(os, 'pyomo'):
        walker = StreamBasedExpressionVisitor(beforeChild=before)
    self.assertIn('Note that the API for the StreamBasedExpressionVisitor has changed to include the child index for the beforeChild() method', os.getvalue().replace('\n', ' '))
    ans = self.walk(walker, self.e)
    m = self.m
    ref = [[[m.x], [2]], [m.y], [[m.z], [[m.x], [m.y]]]]
    self.assertEqual(str(ans), str(ref))
    ans = self.walk(walker, m.x)
    ref = []
    self.assertEqual(str(ans), str(ref))
    ans = self.walk(walker, 2)
    ref = []
    self.assertEqual(str(ans), str(ref))