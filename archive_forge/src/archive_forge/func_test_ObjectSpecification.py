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
def test_ObjectSpecification(self):
    c = C()
    directlyProvides(c, I4)
    self.assertEqual([i.getName() for i in providedBy(c)], ['I4', 'I31', 'I1', 'I2'])
    self.assertEqual([i.getName() for i in providedBy(c).flattened()], ['I4', 'I31', 'I3', 'I1', 'I2', 'Interface'])
    self.assertTrue(I1 in providedBy(c))
    self.assertFalse(I3 in providedBy(c))
    self.assertTrue(providedBy(c).extends(I3))
    self.assertTrue(providedBy(c).extends(I31))
    self.assertFalse(providedBy(c).extends(I5))

    class COnly(A, B):
        pass
    classImplementsOnly(COnly, I31)

    class D(COnly):
        pass
    classImplements(D, I5)
    classImplements(D, I5)
    c = D()
    directlyProvides(c, I4)
    self.assertEqual([i.getName() for i in providedBy(c)], ['I4', 'I5', 'I31'])
    self.assertEqual([i.getName() for i in providedBy(c).flattened()], ['I4', 'I5', 'I31', 'I3', 'Interface'])
    self.assertFalse(I1 in providedBy(c))
    self.assertFalse(I3 in providedBy(c))
    self.assertTrue(providedBy(c).extends(I3))
    self.assertFalse(providedBy(c).extends(I1))
    self.assertTrue(providedBy(c).extends(I31))
    self.assertTrue(providedBy(c).extends(I5))

    class COnly(A, B):
        __implemented__ = I31

    class D(COnly):
        pass
    classImplements(D, I5)
    classImplements(D, I5)
    c = D()
    directlyProvides(c, I4)
    self.assertEqual([i.getName() for i in providedBy(c)], ['I4', 'I5', 'I31'])
    self.assertEqual([i.getName() for i in providedBy(c).flattened()], ['I4', 'I5', 'I31', 'I3', 'Interface'])
    self.assertFalse(I1 in providedBy(c))
    self.assertFalse(I3 in providedBy(c))
    self.assertTrue(providedBy(c).extends(I3))
    self.assertFalse(providedBy(c).extends(I1))
    self.assertTrue(providedBy(c).extends(I31))
    self.assertTrue(providedBy(c).extends(I5))