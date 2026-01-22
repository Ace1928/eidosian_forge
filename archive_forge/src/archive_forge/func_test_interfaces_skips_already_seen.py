import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_interfaces_skips_already_seen(self):
    from zope.interface.interface import Interface

    class IFoo(Interface):
        pass
    spec = self._makeOne([IFoo, IFoo])
    self.assertEqual(list(spec.interfaces()), [IFoo])