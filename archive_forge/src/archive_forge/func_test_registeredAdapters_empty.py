import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registeredAdapters_empty(self):
    comp = self._makeOne()
    self.assertEqual(list(comp.registeredAdapters()), [])