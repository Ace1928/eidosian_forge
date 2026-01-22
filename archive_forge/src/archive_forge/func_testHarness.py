from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testHarness(self):
    xml = b'<root><child/><child2/></root>'
    self.stream.parse(xml)
    self.assertEqual(self.doc_started, True)
    self.assertEqual(self.root.name, 'root')
    self.assertEqual(self.elements[0].name, 'child')
    self.assertEqual(self.elements[1].name, 'child2')
    self.assertEqual(self.doc_ended, True)