import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test___lt___miss(self):
    _component = object()
    _component2 = object()
    ar, _registry, _name = self._makeOne(_component)
    ar2, _, _ = self._makeOne(_component2)
    ar2.name = _name + '2'
    self.assertTrue(ar < ar2)