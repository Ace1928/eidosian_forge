from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def test_sendAuthInitialResponse(self):
    """
        Test starting authentication with an initial response.
        """
    self.init.initialResponse = b'dummy'
    self.init.start()
    auth = self.output[0]
    self.assertEqual(NS_XMPP_SASL, auth.uri)
    self.assertEqual('auth', auth.name)
    self.assertEqual('DUMMY', auth['mechanism'])
    self.assertEqual('ZHVtbXk=', str(auth))