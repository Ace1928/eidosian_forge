import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_ctor_w_base(self):
    base = self._makeOne('base')
    comp = self._makeOne('testing', (base,))
    self.assertEqual(comp.__name__, 'testing')
    self.assertEqual(comp.__bases__, (base,))
    self.assertEqual(comp.adapters.__bases__, (base.adapters,))
    self.assertEqual(comp.utilities.__bases__, (base.utilities,))