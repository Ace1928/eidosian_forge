import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_registerUtility_with_component_name(self):
    from zope.interface.declarations import InterfaceClass
    from zope.interface.declarations import named

    class IFoo(InterfaceClass):
        pass
    ifoo = IFoo('IFoo')

    @named('foo')
    class Foo:
        pass
    foo = Foo()
    _info = 'info'
    comp = self._makeOne()
    comp.registerUtility(foo, ifoo, info=_info)
    self.assertEqual(comp._utility_registrations[ifoo, 'foo'], (foo, _info, None))