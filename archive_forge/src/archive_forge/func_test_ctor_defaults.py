import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_ctor_defaults(self):
    klass = self._getTargetClass()
    inst = klass('ITesting')
    self.assertEqual(inst.__name__, 'ITesting')
    self.assertEqual(inst.__doc__, '')
    self.assertEqual(inst.__bases__, ())
    self.assertEqual(inst.getBases(), ())