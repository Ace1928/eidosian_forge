import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test___add___overlapping_interface(self):
    from zope.interface import Interface
    from zope.interface import ro
    from zope.interface.interface import InterfaceClass
    from zope.interface.tests.test_ro import C3Setting
    IBase = InterfaceClass('IBase')
    IDerived = InterfaceClass('IDerived', (IBase,))
    with C3Setting(ro.C3.STRICT_IRO, True):
        base = self._makeOne(IBase)
        after = base + IDerived
    self.assertEqual(after.__iro__, (IDerived, IBase, Interface))
    self.assertEqual(after.__bases__, (IDerived, IBase))
    self.assertEqual(list(after), [IDerived, IBase])