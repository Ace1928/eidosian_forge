import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___getitem__nonesuch(self):
    from zope.interface import Interface

    class IEmpty(Interface):
        pass
    self.assertRaises(KeyError, IEmpty.__getitem__, 'nonesuch')