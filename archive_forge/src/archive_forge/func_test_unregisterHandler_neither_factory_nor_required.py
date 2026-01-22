import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_unregisterHandler_neither_factory_nor_required(self):
    comp = self._makeOne()
    self.assertRaises(TypeError, comp.unregisterHandler)