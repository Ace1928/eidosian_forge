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
def test_suffix_warning_new_components(self):
    m = ConcreteModel()
    m.junk = Suffix(direction=Suffix.EXPORT)
    m.x = Var()
    m.y = Var()
    m.z = Var([1, 2, 3])
    m.o = Objective(expr=m.x + m.z[2])
    m.c = Constraint(expr=m.y <= 0)
    m.c.deactivate()

    @m.Constraint([1, 2, 3])
    def d(m, i):
        return m.z[i] <= 0
    m.d.deactivate()
    m.d[2].activate()
    m.junk[m.x] = 1
    OUT = io.StringIO()
    with LoggingIntercept() as LOG:
        nl_writer.NLWriter().write(m, OUT)
    self.assertEqual(LOG.getvalue(), '')
    m.junk[m.y] = 1
    with LoggingIntercept() as LOG:
        nl_writer.NLWriter().write(m, OUT)
    self.assertEqual("model contains export suffix 'junk' that contains 1 component keys that are not exported as part of the NL file.  Skipping.\n", LOG.getvalue())
    with LoggingIntercept(level=logging.DEBUG) as LOG:
        nl_writer.NLWriter().write(m, OUT)
    self.assertEqual("model contains export suffix 'junk' that contains 1 component keys that are not exported as part of the NL file.  Skipping.\nSkipped component keys:\n\ty\n", LOG.getvalue())
    m.junk[m.z] = 1
    with LoggingIntercept() as LOG:
        nl_writer.NLWriter().write(m, OUT)
    self.assertEqual("model contains export suffix 'junk' that contains 3 component keys that are not exported as part of the NL file.  Skipping.\n", LOG.getvalue())
    with LoggingIntercept(level=logging.DEBUG) as LOG:
        nl_writer.NLWriter().write(m, OUT)
    self.assertEqual("model contains export suffix 'junk' that contains 3 component keys that are not exported as part of the NL file.  Skipping.\nSkipped component keys:\n\ty\n\tz[1]\n\tz[3]\n", LOG.getvalue())
    m.junk[m.c] = 2
    with LoggingIntercept() as LOG:
        nl_writer.NLWriter().write(m, OUT)
    self.assertEqual("model contains export suffix 'junk' that contains 4 component keys that are not exported as part of the NL file.  Skipping.\n", LOG.getvalue())
    m.junk[m.d] = 2
    with LoggingIntercept() as LOG:
        nl_writer.NLWriter().write(m, OUT)
    self.assertEqual("model contains export suffix 'junk' that contains 6 component keys that are not exported as part of the NL file.  Skipping.\n", LOG.getvalue())
    m.junk[5] = 5
    with LoggingIntercept() as LOG:
        nl_writer.NLWriter().write(m, OUT)
    self.assertEqual("model contains export suffix 'junk' that contains 6 component keys that are not exported as part of the NL file.  Skipping.\nmodel contains export suffix 'junk' that contains 1 keys that are not Var, Constraint, Objective, or the model.  Skipping.\n", LOG.getvalue())
    with LoggingIntercept(level=logging.DEBUG) as LOG:
        nl_writer.NLWriter().write(m, OUT)
    self.assertEqual("model contains export suffix 'junk' that contains 6 component keys that are not exported as part of the NL file.  Skipping.\nSkipped component keys:\n\tc\n\td[1]\n\td[3]\n\ty\n\tz[1]\n\tz[3]\nmodel contains export suffix 'junk' that contains 1 keys that are not Var, Constraint, Objective, or the model.  Skipping.\nSkipped component keys:\n\t5\n", LOG.getvalue())