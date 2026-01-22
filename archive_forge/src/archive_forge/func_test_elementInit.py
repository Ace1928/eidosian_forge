from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def test_elementInit(self):
    """
        Basic L{domish.Element} initialization tests.
        """
    e = domish.Element((None, 'foo'))
    self.assertEqual(e.name, 'foo')
    self.assertEqual(e.uri, None)
    self.assertEqual(e.defaultUri, None)
    self.assertEqual(e.parent, None)
    e = domish.Element(('', 'foo'))
    self.assertEqual(e.name, 'foo')
    self.assertEqual(e.uri, '')
    self.assertEqual(e.defaultUri, '')
    self.assertEqual(e.parent, None)
    e = domish.Element(('testns', 'foo'))
    self.assertEqual(e.name, 'foo')
    self.assertEqual(e.uri, 'testns')
    self.assertEqual(e.defaultUri, 'testns')
    self.assertEqual(e.parent, None)
    e = domish.Element(('testns', 'foo'), 'test2ns')
    self.assertEqual(e.name, 'foo')
    self.assertEqual(e.uri, 'testns')
    self.assertEqual(e.defaultUri, 'test2ns')