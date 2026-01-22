import unittest
from zope.interface.tests import OptimizationTestMixin
def test_register_with_invalid_name(self):
    IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
    registry = self._makeOne()
    with self.assertRaises(ValueError):
        registry.register([IB0], IR0, object(), 'A1')