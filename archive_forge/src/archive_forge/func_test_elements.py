from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def test_elements(self):
    """
        Calling C{elements} without arguments on a L{domish.Element} returns
        all child elements, whatever the qualified name.
        """
    e = domish.Element(('testns', 'foo'))
    c1 = e.addElement('name')
    c2 = e.addElement(('testns2', 'baz'))
    c3 = e.addElement('quux')
    c4 = e.addElement(('testns', 'name'))
    elts = list(e.elements())
    self.assertIn(c1, elts)
    self.assertIn(c2, elts)
    self.assertIn(c3, elts)
    self.assertIn(c4, elts)