import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___call___w_overridden_adapt_and_conform(self):
    from zope.interface import Interface
    from zope.interface import implementer
    from zope.interface import interfacemethod

    class IAdapt(Interface):

        @interfacemethod
        def __adapt__(self, obj):
            return 42

    class ISimple(Interface):
        """Nothing special."""

    @implementer(IAdapt)
    class Conform24:

        def __conform__(self, iface):
            return 24

    @implementer(IAdapt)
    class ConformNone:

        def __conform__(self, iface):
            return None
    self.assertEqual(42, IAdapt(object()))
    self.assertEqual(24, ISimple(Conform24()))
    self.assertEqual(24, IAdapt(Conform24()))
    with self.assertRaises(TypeError):
        ISimple(ConformNone())
    self.assertEqual(42, IAdapt(ConformNone()))