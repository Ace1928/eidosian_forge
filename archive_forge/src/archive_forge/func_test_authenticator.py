from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import defer, task
from twisted.internet.error import ConnectionLost
from twisted.internet.interfaces import IProtocolFactory
from twisted.python import failure
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words.protocols.jabber import error, ijabber, jid, xmlstream
from twisted.words.test.test_xmlstream import GenericXmlStreamFactoryTestsMixin
from twisted.words.xish import domish
def test_authenticator(self):
    """
        Test that the associated authenticator is correctly called.
        """
    connectionMadeCalls = []
    streamStartedCalls = []
    associateWithStreamCalls = []

    class TestAuthenticator:

        def connectionMade(self):
            connectionMadeCalls.append(None)

        def streamStarted(self, rootElement):
            streamStartedCalls.append(rootElement)

        def associateWithStream(self, xs):
            associateWithStreamCalls.append(xs)
    a = TestAuthenticator()
    xs = xmlstream.XmlStream(a)
    self.assertEqual([xs], associateWithStreamCalls)
    xs.connectionMade()
    self.assertEqual([None], connectionMadeCalls)
    xs.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' from='example.com' id='12345'>")
    self.assertEqual(1, len(streamStartedCalls))
    xs.reset()
    self.assertEqual([None], connectionMadeCalls)