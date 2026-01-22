import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_ctor_w_explicit___doc__(self):
    ATTRS = {'__doc__': 'ATTR'}
    klass = self._getTargetClass()
    inst = klass('ITesting', attrs=ATTRS, __doc__='EXPLICIT')
    self.assertEqual(inst.__doc__, 'EXPLICIT')