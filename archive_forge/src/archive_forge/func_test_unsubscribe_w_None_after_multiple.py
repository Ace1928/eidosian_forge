import unittest
from zope.interface.tests import OptimizationTestMixin
def test_unsubscribe_w_None_after_multiple(self):
    IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
    registry = self._makeOne()
    first = object()
    second = object()
    registry.subscribe([IB1], None, first)
    registry.subscribe([IB1], None, second)
    self._check_basic_types_of_subscribers(registry, expected_order=2)
    registry.unsubscribe([IB1], None)
    self.assertEqual(len(registry._subscribers), 0)