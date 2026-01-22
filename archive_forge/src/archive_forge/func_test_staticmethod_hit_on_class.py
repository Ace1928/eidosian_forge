import unittest
def test_staticmethod_hit_on_class(self):
    from zope.interface import Interface
    from zope.interface import provider
    from zope.interface.verify import verifyObject

    class IFoo(Interface):

        def bar(a, b):
            """The bar method"""

    @provider(IFoo)
    class Foo:

        @staticmethod
        def bar(a, b):
            raise AssertionError("We're never actually called")
    verifyObject(IFoo, Foo)