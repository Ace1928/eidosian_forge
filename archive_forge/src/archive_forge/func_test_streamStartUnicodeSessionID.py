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
def test_streamStartUnicodeSessionID(self):
    """
        The generated session id must be a unicode object.
        """
    xs = self.xmlstream
    xs.makeConnection(proto_helpers.StringTransport())
    xs.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' from='example.org' to='example.com' id='12345' version='1.0'>")
    self.assertIsInstance(xs.sid, str)