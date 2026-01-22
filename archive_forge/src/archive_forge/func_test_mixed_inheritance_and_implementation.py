import unittest
def test_mixed_inheritance_and_implementation(self):
    from zope.interface import Interface
    from zope.interface import implementedBy
    from zope.interface import implementer
    from zope.interface import providedBy

    class IFoo(Interface):
        pass

    @implementer(IFoo)
    class ImplementsFoo:
        pass

    class ExtendsFoo(ImplementsFoo):
        pass

    class ImplementsNothing:
        pass

    class ExtendsFooImplementsNothing(ExtendsFoo, ImplementsNothing):
        pass
    self.assertEqual(self._callFUT(providedBy(ExtendsFooImplementsNothing())), [implementedBy(ExtendsFooImplementsNothing), implementedBy(ExtendsFoo), implementedBy(ImplementsFoo), IFoo, Interface, implementedBy(ImplementsNothing), implementedBy(object)])