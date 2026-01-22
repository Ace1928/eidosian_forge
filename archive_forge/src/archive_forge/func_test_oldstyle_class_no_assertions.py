import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_oldstyle_class_no_assertions(self):

    class Foo:
        pass
    self.assertEqual(list(self._callFUT(Foo)), [])