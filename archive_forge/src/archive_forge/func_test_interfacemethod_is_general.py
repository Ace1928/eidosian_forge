import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_interfacemethod_is_general(self):
    from zope.interface import Interface
    from zope.interface import interfacemethod

    class I(Interface):

        @interfacemethod
        def __call__(self, obj):
            """Replace an existing method"""
            return 42

        @interfacemethod
        def this_is_new(self):
            return 42
    self.assertEqual(I(self), 42)
    self.assertEqual(I.this_is_new(), 42)