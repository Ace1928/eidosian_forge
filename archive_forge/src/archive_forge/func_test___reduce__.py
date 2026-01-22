import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___reduce__(self):
    iface = self._makeOne('PickleMe')
    self.assertEqual(iface.__reduce__(), 'PickleMe')