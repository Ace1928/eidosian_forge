import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_proxy_equality(self):

    class Proxy:

        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name):
            raise NotImplementedError()

        def __eq__(self, other):
            return self._wrapped == other

        def __ne__(self, other):
            return self._wrapped != other
    from zope.interface.declarations import implementedBy

    class A:
        pass

    class B:
        pass
    implementedByA = implementedBy(A)
    implementedByB = implementedBy(B)
    proxy = Proxy(implementedByA)
    self.assertTrue(implementedByA == implementedByA)
    self.assertTrue(implementedByA != implementedByB)
    self.assertTrue(implementedByB != implementedByA)
    self.assertTrue(proxy == implementedByA)
    self.assertTrue(implementedByA == proxy)
    self.assertFalse(proxy != implementedByA)
    self.assertFalse(implementedByA != proxy)
    self.assertTrue(proxy != implementedByB)
    self.assertTrue(implementedByB != proxy)