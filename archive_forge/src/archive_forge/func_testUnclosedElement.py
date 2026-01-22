from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def testUnclosedElement(self):
    self.assertRaises(domish.ParserError, self.stream.parse, b'<root><error></root>')