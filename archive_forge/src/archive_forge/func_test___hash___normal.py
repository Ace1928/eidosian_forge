import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___hash___normal(self):
    iface = self._makeOne('HashMe')
    self.assertEqual(hash(iface), hash(('HashMe', 'zope.interface.tests.test_interface')))