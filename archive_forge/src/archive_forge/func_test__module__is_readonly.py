import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test__module__is_readonly(self):
    inst = self._makeOne()
    with self.assertRaises(AttributeError):
        inst.__module__ = 'different.module'