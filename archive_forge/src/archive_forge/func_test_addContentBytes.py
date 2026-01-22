from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def test_addContentBytes(self):
    """
        Byte strings passed to C{addContent} are not acceptable on Python 3.
        """
    element = domish.Element(('testns', 'foo'))
    self.assertRaises(TypeError, element.addContent, b'bytes')