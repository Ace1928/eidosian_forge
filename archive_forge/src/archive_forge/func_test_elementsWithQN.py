from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def test_elementsWithQN(self):
    """
        Calling C{elements} with a namespace and local name on a
        L{domish.Element} returns all child elements with that qualified name.
        """
    e = domish.Element(('testns', 'foo'))
    c1 = e.addElement('name')
    c2 = e.addElement(('testns2', 'baz'))
    c3 = e.addElement('quux')
    c4 = e.addElement(('testns', 'name'))
    elts = list(e.elements('testns', 'name'))
    self.assertIn(c1, elts)
    self.assertNotIn(c2, elts)
    self.assertNotIn(c3, elts)
    self.assertIn(c4, elts)