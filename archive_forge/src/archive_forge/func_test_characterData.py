from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def test_characterData(self):
    """
        Extract character data using L{str}.
        """
    element = domish.Element(('testns', 'foo'))
    element.addContent('somecontent')
    text = str(element)
    self.assertEqual('somecontent', text)
    self.assertIsInstance(text, str)