import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___call___miss_w_alternate(self):
    from zope.interface import Interface

    class I(Interface):
        pass

    class C:
        pass
    c = C()
    self.assertTrue(I(c, self) is self)