import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___call___wo___conform___ob_no_provides_w_alternate(self):
    ib = self._makeOne(False)
    __traceback_info__ = (ib, self._getTargetClass())
    adapted = object()
    alternate = object()
    self.assertIs(ib(adapted, alternate), alternate)