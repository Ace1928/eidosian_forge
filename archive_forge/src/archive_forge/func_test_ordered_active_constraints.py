import logging
import math
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import (
from pyomo.environ import (
import pyomo.repn.util
from pyomo.repn.util import (
def test_ordered_active_constraints(self):

    class MockConfig(object):
        row_order = None
        file_determinism = FileDeterminism(0)
    m = ConcreteModel()
    m.v = Var()
    m.x = Constraint(expr=m.v >= 0)
    m.y = Constraint([3, 2], rule=lambda b, i: m.v >= 0)
    m.c = Block()
    m.c.x = Constraint(expr=m.v >= 0)
    m.c.y = Constraint([5, 4], rule=lambda b, i: m.v >= 0)
    m.b = Block()
    m.b.x = Constraint(expr=m.v >= 0)
    m.b.y = Constraint([7, 6], rule=lambda b, i: m.v >= 0)
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.x, m.y[3], m.y[2], m.c.x, m.c.y[5], m.c.y[4], m.b.x, m.b.y[7], m.b.y[6]])
    MockConfig.file_determinism = FileDeterminism.SORT_INDICES
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.x, m.y[2], m.y[3], m.c.x, m.c.y[4], m.c.y[5], m.b.x, m.b.y[6], m.b.y[7]])
    MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.x, m.y[2], m.y[3], m.b.x, m.b.y[6], m.b.y[7], m.c.x, m.c.y[4], m.c.y[5]])
    MockConfig.row_order = []
    MockConfig.file_determinism = FileDeterminism(0)
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.x, m.y[3], m.y[2], m.c.x, m.c.y[5], m.c.y[4], m.b.x, m.b.y[7], m.b.y[6]])
    MockConfig.row_order = False
    MockConfig.file_determinism = FileDeterminism(0)
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.x, m.y[3], m.y[2], m.c.x, m.c.y[5], m.c.y[4], m.b.x, m.b.y[7], m.b.y[6]])
    MockConfig.file_determinism = FileDeterminism.SORT_INDICES
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.x, m.y[2], m.y[3], m.c.x, m.c.y[4], m.c.y[5], m.b.x, m.b.y[6], m.b.y[7]])
    MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.x, m.y[2], m.y[3], m.b.x, m.b.y[6], m.b.y[7], m.c.x, m.c.y[4], m.c.y[5]])
    MockConfig.row_order = True
    MockConfig.file_determinism = FileDeterminism(0)
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.x, m.y[3], m.y[2], m.c.x, m.c.y[5], m.c.y[4], m.b.x, m.b.y[7], m.b.y[6]])
    MockConfig.row_order = True
    MockConfig.file_determinism = FileDeterminism.SORT_INDICES
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.x, m.y[2], m.y[3], m.c.x, m.c.y[4], m.c.y[5], m.b.x, m.b.y[6], m.b.y[7]])
    MockConfig.row_order = True
    MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.x, m.y[2], m.y[3], m.b.x, m.b.y[6], m.b.y[7], m.c.x, m.c.y[4], m.c.y[5]])
    MockConfig.row_order = ComponentMap(((v, i) for i, v in enumerate([m.b.y, m.y, m.c.y[4], m.x])))
    MockConfig.file_determinism = FileDeterminism.ORDERED
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x])
    MockConfig.file_determinism = FileDeterminism.SORT_INDICES
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x])
    MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]])
    MockConfig.row_order = [m.b.y, m.y, m.c.y[4], m.x]
    ref = list(MockConfig.row_order)
    MockConfig.file_determinism = FileDeterminism.ORDERED
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x])
    self.assertEqual(MockConfig.row_order, ref)
    MockConfig.file_determinism = FileDeterminism.SORT_INDICES
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x])
    self.assertEqual(MockConfig.row_order, ref)
    MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
    self.assertEqual(list(ordered_active_constraints(m, MockConfig)), [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]])
    self.assertEqual(MockConfig.row_order, ref)