import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_super_when_parent_implements_interface_diamond(self):
    from zope.interface import Interface
    from zope.interface.declarations import implementer

    class IBase(Interface):
        pass

    class IDerived(IBase):
        pass

    class Base:
        pass

    class Child1(Base):
        pass

    @implementer(IBase)
    class Child2(Base):
        pass

    @implementer(IDerived)
    class Derived(Child1, Child2):
        pass
    self.assertEqual(Derived.__mro__, (Derived, Child1, Child2, Base, object))
    self.assertEqual(list(self._callFUT(Derived)), [IDerived, IBase])
    sup = super(Derived, Derived)
    fut = self._callFUT(sup)
    self.assertEqual(list(fut), [IBase])
    self.assertIsNone(fut._dependents)