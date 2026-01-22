import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_getAdapter_hit_super_when_parent_implements_interface_diamond(self):
    from zope.interface import Interface
    from zope.interface.declarations import implementer

    class IBase(Interface):
        pass

    class IDerived(IBase):
        pass

    class IFoo(Interface):
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

    class AdapterBase:

        def __init__(self, context):
            self.context = context

    class AdapterDerived:

        def __init__(self, context):
            self.context = context
    comp = self._makeOne()
    comp.registerAdapter(AdapterDerived, (IDerived,), IFoo)
    comp.registerAdapter(AdapterBase, (IBase,), IFoo)
    self._should_not_change(comp)
    derived = Derived()
    adapter = comp.getAdapter(derived, IFoo)
    self.assertIsInstance(adapter, AdapterDerived)
    self.assertIs(adapter.context, derived)
    supe = super(Derived, derived)
    adapter = comp.getAdapter(supe, IFoo)
    self.assertIsInstance(adapter, AdapterBase)
    self.assertIs(adapter.context, derived)