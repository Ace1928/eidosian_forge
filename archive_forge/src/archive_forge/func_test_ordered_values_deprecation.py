import os
from os.path import abspath, dirname
from pyomo.common import DeveloperError
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Var, Param, Set, value, Integers
from pyomo.core.base.set import FiniteSetOf, OrderedSetOf
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.expr import GetItemExpression
from pyomo.core import SortComponents
def test_ordered_values_deprecation(self):
    m = ConcreteModel()
    unordered = [1, 3, 2]
    ordered = [1, 2, 3]
    m.I = FiniteSetOf(unordered)
    m.x = Var(m.I)
    unordered = [m.x[i] for i in unordered]
    ordered = [m.x[i] for i in ordered]
    self.assertEqual(list(m.x.values()), unordered)
    self.assertEqual(list(m.x.values(SortComponents.ORDERED_INDICES)), ordered)
    with LoggingIntercept() as LOG:
        self.assertEqual(list(m.x.values(True)), ordered)
    self.assertEqual(LOG.getvalue(), '')
    with LoggingIntercept() as LOG:
        self.assertEqual(list(m.x.values(ordered=True)), ordered)
    self.assertIn('values(ordered=True) is deprecated', LOG.getvalue())
    with LoggingIntercept() as LOG:
        self.assertEqual(list(m.x.values(ordered=False)), unordered)
    self.assertIn('values(ordered=False) is deprecated', LOG.getvalue())
    m = ConcreteModel()
    unordered = [1, 3, 2]
    ordered = [1, 2, 3]
    m.I = OrderedSetOf(unordered)
    m.x = Var(m.I)
    unordered = [m.x[i] for i in unordered]
    ordered = [m.x[i] for i in ordered]
    self.assertEqual(list(m.x.values()), unordered)
    self.assertEqual(list(m.x.values(SortComponents.ORDERED_INDICES)), unordered)
    with LoggingIntercept() as LOG:
        self.assertEqual(list(m.x.values(True)), ordered)
    self.assertEqual(LOG.getvalue(), '')
    with LoggingIntercept() as LOG:
        self.assertEqual(list(m.x.values(ordered=True)), unordered)
    self.assertIn('values(ordered=True) is deprecated', LOG.getvalue())
    with LoggingIntercept() as LOG:
        self.assertEqual(list(m.x.values(ordered=False)), unordered)
    self.assertIn('values(ordered=False) is deprecated', LOG.getvalue())