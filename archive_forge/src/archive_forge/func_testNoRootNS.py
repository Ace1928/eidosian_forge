from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testNoRootNS(self):
    xml = b"<stream><error xmlns='etherx'/></stream>"
    self.stream.parse(xml)
    self.assertEqual(self.root.uri, '')
    self.assertEqual(self.elements[0].uri, 'etherx')