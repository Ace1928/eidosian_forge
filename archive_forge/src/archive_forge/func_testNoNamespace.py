from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testNoNamespace(self):
    e = domish.Element((None, 'foo'))
    self.assertEqual(e.toXml(), '<foo/>')
    self.assertEqual(e.toXml(closeElement=0), '<foo>')