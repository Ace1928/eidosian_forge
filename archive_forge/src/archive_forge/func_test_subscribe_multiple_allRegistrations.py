import unittest
from zope.interface.tests import OptimizationTestMixin
def test_subscribe_multiple_allRegistrations(self):
    IB0, IB1, IB2, IB3, IB4, IF0, IF1, IR0, IR1 = _makeInterfaces()
    registry = self._makeOne()
    registry.subscribe([], IR0, 'A1')
    registry.subscribe([], IR0, 'A2')
    registry.subscribe([IB0], IR0, 'A1')
    registry.subscribe([IB0], IR0, 'A2')
    registry.subscribe([IB0], IR1, 'A3')
    registry.subscribe([IB0], IR1, 'A4')
    registry.subscribe([IB0, IB1], IR0, 'A1')
    registry.subscribe([IB0, IB2], IR0, 'A2')
    registry.subscribe([IB0, IB2], IR1, 'A4')
    registry.subscribe([IB0, IB3], IR1, 'A3')

    def build_subscribers(L, F, MT):
        return L([MT({IR0: MT({'': F(['A1', 'A2'])})}), MT({IB0: MT({IR0: MT({'': F(['A1', 'A2'])}), IR1: MT({'': F(['A3', 'A4'])})})}), MT({IB0: MT({IB1: MT({IR0: MT({'': F(['A1'])})}), IB2: MT({IR0: MT({'': F(['A2'])}), IR1: MT({'': F(['A4'])})}), IB3: MT({IR1: MT({'': F(['A3'])})})})})])
    self.assertEqual(registry._subscribers, build_subscribers(L=self._getMutableListType(), F=self._getLeafSequenceType(), MT=self._getMappingType()))

    def build_provided(P):
        return P({IR0: 6, IR1: 4})
    self.assertEqual(registry._provided, build_provided(P=self._getProvidedType()))
    registered = sorted(registry.allSubscriptions())
    self.assertEqual(registered, [((), IR0, 'A1'), ((), IR0, 'A2'), ((IB0,), IR0, 'A1'), ((IB0,), IR0, 'A2'), ((IB0,), IR1, 'A3'), ((IB0,), IR1, 'A4'), ((IB0, IB1), IR0, 'A1'), ((IB0, IB2), IR0, 'A2'), ((IB0, IB2), IR1, 'A4'), ((IB0, IB3), IR1, 'A3')])
    registry2 = self._makeOne()
    for args in registered:
        registry2.subscribe(*args)
    self.assertEqual(registry2._subscribers, registry._subscribers)
    self.assertEqual(registry2._provided, registry._provided)
    registry._mappingType = CustomMapping
    registry._leafSequenceType = CustomLeafSequence
    registry._sequenceType = CustomSequence
    registry._providedType = CustomProvided

    def addValue(existing, new):
        existing = existing if existing is not None else CustomLeafSequence()
        existing.append(new)
        return existing
    registry._addValueToLeaf = addValue
    registry.rebuild()
    self.assertEqual(registry._subscribers, build_subscribers(L=CustomSequence, F=CustomLeafSequence, MT=CustomMapping))