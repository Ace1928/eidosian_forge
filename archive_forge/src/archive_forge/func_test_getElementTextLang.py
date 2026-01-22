from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
def test_getElementTextLang(self) -> None:
    """
        Test getting an element for an error with a text and language.
        """
    e = error.BaseError('feature-not-implemented', 'text', 'en_US')
    element = e.getElement()
    self.assertEqual(len(element.children), 2)
    self.assertEqual(str(element.text), 'text')
    self.assertEqual(element.text[NS_XML, 'lang'], 'en_US')