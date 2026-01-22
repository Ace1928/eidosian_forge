import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_getDescriptionFor_nonesuch_no_default(self):
    from zope.interface import Interface

    class IEmpty(Interface):
        pass
    self.assertRaises(KeyError, IEmpty.getDescriptionFor, 'nonesuch')