import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___adapt___as_method_and_implementation(self):
    from zope.interface import Interface
    from zope.interface import interfacemethod

    class I(Interface):

        @interfacemethod
        def __adapt__(self, obj):
            return 42

        def __adapt__(to_adapt):
            """This is a protocol"""
    self.assertEqual(42, I(object()))
    self.assertEqual(I['__adapt__'].getSignatureString(), '(to_adapt)')