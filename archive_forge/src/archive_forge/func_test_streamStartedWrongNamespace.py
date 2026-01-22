from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
def test_streamStartedWrongNamespace(self):
    """
        The received stream header should have a correct namespace.
        """
    streamErrors = []
    xs = self.xmlstream
    xs.sendStreamError = streamErrors.append
    xs.makeConnection(self)
    xs.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' to='component.example.org'>")
    self.assertEqual(1, len(streamErrors))
    self.assertEqual('invalid-namespace', streamErrors[-1].condition)