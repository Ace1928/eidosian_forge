import unittest
from zope.interface.tests import OptimizationTestMixin
def test_unsubscribe_after_multiple(self):
    IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
    registry = self._makeOne()
    first = object()
    second = object()
    third = object()
    fourth = object()
    registry.subscribe([IB1], None, first)
    registry.subscribe([IB1], None, second)
    registry.subscribe([IB1], IR0, third)
    registry.subscribe([IB1], IR0, fourth)
    self._check_basic_types_of_subscribers(registry, expected_order=2)
    MT = self._getMappingType()
    L = self._getLeafSequenceType()
    PT = self._getProvidedType()
    self.assertEqual(registry._subscribers[1], MT({IB1: MT({None: MT({'': L((first, second))}), IR0: MT({'': L((third, fourth))})})}))
    self.assertEqual(registry._provided, PT({IR0: 2}))
    IR0_leaf_orig = registry._subscribers[1][IB1][IR0]['']
    Non_leaf_orig = registry._subscribers[1][IB1][None]['']
    registry.unsubscribe([IB1], None, first)
    registry.unsubscribe([IB1], IR0, third)
    self.assertEqual(registry._subscribers[1], MT({IB1: MT({None: MT({'': L((second,))}), IR0: MT({'': L((fourth,))})})}))
    self.assertEqual(registry._provided, PT({IR0: 1}))
    IR0_leaf_new = registry._subscribers[1][IB1][IR0]['']
    Non_leaf_new = registry._subscribers[1][IB1][None]['']
    self.assertLeafIdentity(IR0_leaf_orig, IR0_leaf_new)
    self.assertLeafIdentity(Non_leaf_orig, Non_leaf_new)
    registry.unsubscribe([IB1], None, second)
    registry.unsubscribe([IB1], IR0, fourth)
    self.assertEqual(len(registry._subscribers), 0)
    self.assertEqual(len(registry._provided), 0)