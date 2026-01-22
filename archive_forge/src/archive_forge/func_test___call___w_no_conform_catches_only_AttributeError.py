import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___call___w_no_conform_catches_only_AttributeError(self):
    MissingSomeAttrs.test_raises(self, self._makeOne(), expected_missing='__conform__')