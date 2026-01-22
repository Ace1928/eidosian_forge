from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testChildInDefaultNamespace(self):
    e = domish.Element(('testns', 'foo'), 'testns2')
    e.addElement(('testns2', 'bar'))
    self.assertEqual(e.toXml(), "<xn0:foo xmlns:xn0='testns' xmlns='testns2'><bar/></xn0:foo>")