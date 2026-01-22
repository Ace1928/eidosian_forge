import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test___eq___identity(self):
    _component = object()
    ar, _registry, _name = self._makeOne(_component)
    self.assertTrue(ar == ar)