import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_attributes_link_to_interface(self):
    from zope.interface import Attribute
    from zope.interface import Interface

    class I1(Interface):
        attr = Attribute('My attr')
    self.assertTrue(I1['attr'].interface is I1)