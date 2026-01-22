import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_unregisterUtility_both_factory_and_component(self):

    def _factory():
        raise NotImplementedError()
    _to_reg = object()
    comp = self._makeOne()
    self.assertRaises(TypeError, comp.unregisterUtility, component=_to_reg, factory=_factory)