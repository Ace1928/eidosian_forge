import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_methods_link_to_interface(self):
    from zope.interface import Interface

    class I1(Interface):

        def method(foo, bar, bingo):
            """A method"""
    self.assertTrue(I1['method'].interface is I1)