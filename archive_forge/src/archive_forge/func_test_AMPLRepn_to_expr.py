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
def test_AMPLRepn_to_expr(self):
    m = ConcreteModel()
    m.p = Param([2, 3, 4], mutable=True, initialize=lambda m, i: i ** 2)
    m.x = Var([2, 3, 4], initialize=lambda m, i: i)
    e = 10
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((e, None, None, 1))
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 10)
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
    ee = repn.to_expr(info.var_map)
    self.assertExpressionsEqual(ee, 10)
    e += sum((m.x[i] * m.p[i] for i in m.x))
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((e, None, None, 1))
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 10)
    self.assertEqual(repn.linear, {id(m.x[2]): 4, id(m.x[3]): 9, id(m.x[4]): 16})
    self.assertEqual(repn.nonlinear, None)
    ee = repn.to_expr(info.var_map)
    self.assertExpressionsEqual(ee, 4 * m.x[2] + 9 * m.x[3] + 16 * m.x[4] + 10)
    self.assertEqual(ee(), 10 + 8 + 27 + 64)
    e = sum((m.x[i] * m.p[i] for i in m.x))
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((e, None, None, 1))
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 0)
    self.assertEqual(repn.linear, {id(m.x[2]): 4, id(m.x[3]): 9, id(m.x[4]): 16})
    self.assertEqual(repn.nonlinear, None)
    ee = repn.to_expr(info.var_map)
    self.assertExpressionsEqual(ee, 4 * m.x[2] + 9 * m.x[3] + 16 * m.x[4])
    self.assertEqual(ee(), 8 + 27 + 64)
    e += m.x[2] ** 2
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((e, None, None, 1))
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 0)
    self.assertEqual(repn.linear, {id(m.x[2]): 4, id(m.x[3]): 9, id(m.x[4]): 16})
    self.assertEqual(repn.nonlinear, ('o5\n%s\nn2\n', [id(m.x[2])]))
    with self.assertRaisesRegex(MouseTrap, 'Cannot convert nonlinear AMPLRepn to Pyomo Expression'):
        ee = repn.to_expr(info.var_map)