import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___call___defers_to___conform___(self):
    from zope.interface import Interface
    from zope.interface import implementer

    class I(Interface):
        pass

    @implementer(I)
    class C:

        def __conform__(self, proto):
            return 0
    self.assertEqual(I(C()), 0)