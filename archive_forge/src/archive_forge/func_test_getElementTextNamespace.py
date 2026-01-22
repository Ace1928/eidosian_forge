from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
def test_getElementTextNamespace(self) -> None:
    """
        Test that the error text element has the correct namespace.
        """
    e = error.StanzaError('feature-not-implemented', text='text')
    element = e.getElement()
    self.assertEqual(NS_XMPP_STANZAS, element.text.uri)