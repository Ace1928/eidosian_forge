from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
def onAuthGet(iq):
    """
            Called when the initializer sent a query for authentication methods.

            The response informs the client that plain-text authentication
            is supported.
            """
    response = xmlstream.toResponse(iq, 'result')
    response.addElement(('jabber:iq:auth', 'query'))
    response.query.addElement('username')
    response.query.addElement('password')
    response.query.addElement('resource')
    d = self.waitFor(IQ_AUTH_SET, onAuthSet)
    self.pipe.source.send(response)
    return d