from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testChildPrefix(self):
    xml = b"<root xmlns='testns' xmlns:foo='testns2'><foo:child/></root>"
    self.stream.parse(xml)
    self.assertEqual(self.root.localPrefixes['foo'], 'testns2')
    self.assertEqual(self.elements[0].uri, 'testns2')