from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def test_onChallengeResponse(self):
    """
        A non-empty response gets encoded and included as character data.
        """
    d = self.init.start()
    challenge = domish.Element((NS_XMPP_SASL, 'challenge'))
    challenge.addContent('bXkgY2hhbGxlbmdl')
    self.init.mechanism.response = b'response'
    self.init.onChallenge(challenge)
    response = self.output[1]
    self.assertEqual('cmVzcG9uc2U=', str(response))
    self.init.onSuccess(None)
    return d