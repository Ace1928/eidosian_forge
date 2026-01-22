from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
def test_getElementAppCondition(self) -> None:
    """
        Test getting an element for an error with an app specific condition.
        """
    ac = domish.Element(('testns', 'myerror'))
    e = error.BaseError('feature-not-implemented', appCondition=ac)
    element = e.getElement()
    self.assertEqual(len(element.children), 2)
    self.assertEqual(element.myerror, ac)