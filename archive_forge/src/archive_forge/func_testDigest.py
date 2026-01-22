from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
def testDigest(self):
    """
        Test digest authentication.

        Act as a server supporting digest authentication and expect the
        C{digest} field to be filled with a sha1 digest of the concatenated
        stream session identifier and password. Then act as if authentication
        succeeds.
        """

    def onAuthGet(iq):
        """
            Called when the initializer sent a query for authentication methods.

            The response informs the client that digest authentication is
            supported.
            """
        response = xmlstream.toResponse(iq, 'result')
        response.addElement(('jabber:iq:auth', 'query'))
        response.query.addElement('username')
        response.query.addElement('digest')
        response.query.addElement('resource')
        d = self.waitFor(IQ_AUTH_SET, onAuthSet)
        self.pipe.source.send(response)
        return d

    def onAuthSet(iq):
        """
            Called when the initializer sent the authentication request.

            The server checks the credentials and responds with an empty result
            signalling success.
            """
        self.assertEqual('user', str(iq.query.username))
        self.assertEqual(sha1(b'12345secret').hexdigest(), str(iq.query.digest))
        self.assertEqual('resource', str(iq.query.resource))
        response = xmlstream.toResponse(iq, 'result')
        self.pipe.source.send(response)
    self.xmlstream.sid = '12345'
    d1 = self.waitFor(IQ_AUTH_GET, onAuthGet)
    d2 = self.init.initialize()
    return defer.gatherResults([d1, d2])