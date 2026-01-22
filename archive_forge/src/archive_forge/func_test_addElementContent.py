from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def test_addElementContent(self):
    """
        Content passed to addElement becomes character data on the new child.
        """
    element = domish.Element(('testns', 'foo'))
    child = element.addElement('bar', content='abc')
    self.assertEqual('abc', str(child))