import unittest
from zope.interface.tests import OptimizationTestMixin
def test_register_with_value_None_unregisters(self):
    IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
    registry = self._makeOne()
    registry.register([None], IR0, '', 'A1')
    registry.register([None], IR0, '', None)
    self.assertEqual(len(registry._adapters), 0)
    self.assertIsInstance(registry._adapters, self._getMutableListType())
    registered = list(registry.allRegistrations())
    self.assertEqual(registered, [])