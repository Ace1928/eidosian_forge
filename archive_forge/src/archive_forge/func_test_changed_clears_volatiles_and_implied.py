import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_changed_clears_volatiles_and_implied(self):
    from zope.interface.interface import Interface

    class I(Interface):
        pass
    spec = self._makeOne()
    spec._v_attrs = 'Foo'
    spec._implied[I] = ()
    spec.changed(spec)
    self.assertIsNone(spec._v_attrs)
    self.assertFalse(I in spec._implied)