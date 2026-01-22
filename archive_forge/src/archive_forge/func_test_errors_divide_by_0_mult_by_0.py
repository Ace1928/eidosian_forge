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
def test_errors_divide_by_0_mult_by_0(self):
    m = ConcreteModel()
    m.p = Param(mutable=True, initialize=0)
    m.x = Var()
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((m.p * (1 / m.p), None, None, 1))
    self.assertIn("Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: 1/p\n", LOG.getvalue())
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 0)
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((1 / m.p * m.p, None, None, 1))
    self.assertIn("Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: 1/p\n", LOG.getvalue())
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 0)
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((m.p * (m.x / m.p), None, None, 1))
    self.assertIn("Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: 1/p\n", LOG.getvalue())
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 0)
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((m.p * (3 * (m.x + 2) / m.p), None, None, 1))
    self.assertIn("Exception encountered evaluating expression 'div(3, 0)'\n\tmessage: division by zero\n\texpression: 3*(x + 2)/p\n", LOG.getvalue())
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 0)
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)
    info = INFO()
    with LoggingIntercept() as LOG:
        repn = info.visitor.walk_expression((m.p * (m.x ** 2 / m.p), None, None, 1))
    self.assertIn("Exception encountered evaluating expression 'div(1, 0)'\n\tmessage: division by zero\n\texpression: x**2/p\n", LOG.getvalue())
    self.assertEqual(repn.nl, None)
    self.assertEqual(repn.mult, 1)
    self.assertEqual(repn.const, 0)
    self.assertEqual(repn.linear, {})
    self.assertEqual(repn.nonlinear, None)