import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registeredSubscriptionAdapters_empty(self):
    comp = self._makeOne()
    self.assertEqual(list(comp.registeredSubscriptionAdapters()), [])