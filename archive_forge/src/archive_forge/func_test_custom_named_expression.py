import pyomo.common.unittest as unittest
import io
import logging
import math
import os
import re
import pyomo.repn.util as repn_util
import pyomo.repn.plugins.nl_writer as nl_writer
from pyomo.repn.util import InvalidNumber
from pyomo.repn.tests.nl_diff import nl_diff
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.errors import MouseTrap
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import report_timing
from pyomo.core.expr import Expr_if, inequality, LinearExpression
from pyomo.core.base.expression import ScalarExpression
from pyomo.environ import (
import pyomo.environ as pyo
def test_custom_named_expression(self):

    class CustomExpression(ScalarExpression):
        pass
    m = ConcreteModel()
    m.x = Var()
    m.e = CustomExpression()
    m.e.expr = m.x + 3
    expr = m.e + m.e
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((expr, None, None, 1))
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 6)
    self.assertEqual(repn.linear, {id(m.x): 2})
    self.assertEqual(repn.nonlinear, None)
    self.assertEqual(len(info.subexpression_cache), 1)
    obj, repn, info = info.subexpression_cache[id(m.e)]
    self.assertIs(obj, m.e)
    self.assertEqual(repn.nl, ('%s\n', (id(m.e),)))
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 3)
    self.assertEqual(repn.linear, {id(m.x): 1})
    self.assertEqual(repn.nonlinear, None)
    self.assertEqual(info, [None, None, False])