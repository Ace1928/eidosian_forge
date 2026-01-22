from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
def test_toResponse(self) -> None:
    """
        Test an error response is generated from a stanza.

        The addressing on the (new) response stanza should be reversed, an
        error child (with proper properties) added and the type set to
        C{'error'}.
        """
    stanza = domish.Element(('jabber:client', 'message'))
    stanza['type'] = 'chat'
    stanza['to'] = 'user1@example.com'
    stanza['from'] = 'user2@example.com/resource'
    e = error.StanzaError('service-unavailable')
    response = e.toResponse(stanza)
    self.assertNotIdentical(response, stanza)
    self.assertEqual(response['from'], 'user1@example.com')
    self.assertEqual(response['to'], 'user2@example.com/resource')
    self.assertEqual(response['type'], 'error')
    self.assertEqual(response.error.children[0].name, 'service-unavailable')
    self.assertEqual(response.error['type'], 'cancel')
    self.assertNotEqual(stanza.children, response.children)