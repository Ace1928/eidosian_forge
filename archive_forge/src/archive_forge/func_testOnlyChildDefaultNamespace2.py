from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testOnlyChildDefaultNamespace2(self):
    e = domish.Element((None, 'foo'))
    e.addElement('bar')
    self.assertEqual(e.toXml(), '<foo><bar/></foo>')