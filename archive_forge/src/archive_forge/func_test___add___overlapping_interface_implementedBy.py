import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test___add___overlapping_interface_implementedBy(self):
    from zope.interface import Interface
    from zope.interface import implementedBy
    from zope.interface import implementer
    from zope.interface import ro
    from zope.interface.tests.test_ro import C3Setting

    class IBase(Interface):
        pass

    class IDerived(IBase):
        pass

    @implementer(IBase)
    class Base:
        pass
    with C3Setting(ro.C3.STRICT_IRO, True):
        after = implementedBy(Base) + IDerived
    self.assertEqual(after.__sro__, (after, IDerived, IBase, Interface))
    self.assertEqual(after.__bases__, (IDerived, IBase))
    self.assertEqual(list(after), [IDerived, IBase])