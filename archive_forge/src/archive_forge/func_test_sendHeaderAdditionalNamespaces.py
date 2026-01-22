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
def test_sendHeaderAdditionalNamespaces(self):
    """
        Test for additional namespace declarations.
        """
    xs = self.xmlstream
    xs.prefixes['jabber:server:dialback'] = 'db'
    xs.sendHeader()
    splitHeader = self.xmlstream.transport.value()[0:-1].split(b' ')
    self.assertIn(b'<stream:stream', splitHeader)
    self.assertIn(b"xmlns:stream='http://etherx.jabber.org/streams'", splitHeader)
    self.assertIn(b"xmlns:db='jabber:server:dialback'", splitHeader)
    self.assertIn(b"xmlns='testns'", splitHeader)
    self.assertIn(b"version='1.0'", splitHeader)
    self.assertTrue(xs._headerSent)