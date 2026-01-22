import unittest
from zope.interface import Interface
from zope.interface import classImplements
from zope.interface import classImplementsOnly
from zope.interface import directlyProvidedBy
from zope.interface import directlyProvides
from zope.interface import implementedBy
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface.tests import odd
def test_implementedBy(self):

    class I2(I1):
        pass

    class C1(Odd):
        pass
    classImplements(C1, I2)

    class C2(C1):
        pass
    classImplements(C2, I3)
    self.assertEqual([i.getName() for i in implementedBy(C2)], ['I3', 'I2'])