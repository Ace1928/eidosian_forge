import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_providedBy_miss(self):
    from zope.interface import interface
    from zope.interface.declarations import _empty
    sb = self._makeOne()

    def _providedBy(obj):
        return _empty
    with _Monkey(interface, providedBy=_providedBy):
        self.assertFalse(sb.providedBy(object()))