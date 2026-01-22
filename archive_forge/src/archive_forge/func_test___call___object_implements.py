import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___call___object_implements(self):
    from zope.interface import Interface
    from zope.interface import implementer

    class I(Interface):
        pass

    @implementer(I)
    class C:
        pass
    c = C()
    self.assertTrue(I(c) is c)