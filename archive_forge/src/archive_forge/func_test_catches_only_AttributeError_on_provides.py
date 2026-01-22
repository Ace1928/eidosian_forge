import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_catches_only_AttributeError_on_provides(self):
    MissingSomeAttrs.test_raises(self, self._callFUT, expected_missing='__provides__')