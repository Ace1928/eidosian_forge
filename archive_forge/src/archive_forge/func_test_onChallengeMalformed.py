from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def test_onChallengeMalformed(self):
    """
        Test receiving a malformed challenge message.
        """
    d = self.init.start()
    challenge = domish.Element((NS_XMPP_SASL, 'challenge'))
    challenge.addContent('a')
    self.init.onChallenge(challenge)
    self.assertFailure(d, sasl.SASLIncorrectEncodingError)
    return d