import unittest
from zope.interface.tests import OptimizationTestMixin
def test_register_with_same_value(self):
    from zope.interface import Interface
    IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
    registry = self._makeOne()
    _value = object()
    registry.register([None], IR0, '', _value)
    _before = registry._generation
    registry.register([None], IR0, '', _value)
    self.assertEqual(registry._generation, _before)
    self._check_basic_types_of_adapters(registry)
    MT = self._getMappingType()
    self.assertEqual(registry._adapters[1], MT({Interface: MT({IR0: MT({'': _value})})}))
    registered = list(registry.allRegistrations())
    self.assertEqual(registered, [((Interface,), IR0, '', _value)])