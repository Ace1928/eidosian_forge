from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def test_characterDataMixed(self):
    """
        Mixing addChild with cdata and element, the first cdata is returned.
        """
    element = domish.Element(('testns', 'foo'))
    element.addChild('abc')
    element.addElement('bar')
    element.addChild('def')
    self.assertEqual('abc', str(element))