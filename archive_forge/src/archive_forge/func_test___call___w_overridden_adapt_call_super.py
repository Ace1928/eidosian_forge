import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___call___w_overridden_adapt_call_super(self):
    import sys
    from zope.interface import Interface
    from zope.interface import implementer
    from zope.interface import interfacemethod

    class I(Interface):

        @interfacemethod
        def __adapt__(self, obj):
            if not self.providedBy(obj):
                return 42
            return super().__adapt__(obj)

    @implementer(I)
    class O:
        pass
    self.assertEqual(42, I(object()))
    o = O()
    self.assertIs(o, I(o))