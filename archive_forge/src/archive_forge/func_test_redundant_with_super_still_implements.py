import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_redundant_with_super_still_implements(self):
    Base, IBase = self._check_implementer(type('Foo', (object,), {}), inherit=None)

    class Child(Base):
        pass
    self._callFUT(Child, IBase)
    self.assertTrue(IBase.implementedBy(Child))