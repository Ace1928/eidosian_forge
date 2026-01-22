from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
def onAuthSet(iq):
    """
            Called when the initializer sent the authentication request.

            The server checks the credentials and responds with a not-authorized
            stanza error.
            """
    response = error.StanzaError('not-authorized').toResponse(iq)
    self.pipe.source.send(response)