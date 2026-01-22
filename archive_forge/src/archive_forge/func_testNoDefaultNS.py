from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testNoDefaultNS(self):
    xml = b"<stream:stream xmlns:stream='etherx'><error/></stream:stream>"
    self.stream.parse(xml)
    self.assertEqual(self.root.uri, 'etherx')
    self.assertEqual(self.root.defaultUri, '')
    self.assertEqual(self.elements[0].uri, '')
    self.assertEqual(self.elements[0].defaultUri, '')