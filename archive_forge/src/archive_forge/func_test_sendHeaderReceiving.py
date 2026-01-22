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
def test_sendHeaderReceiving(self):
    """
        Test addressing when receiving a stream.
        """
    xs = self.xmlstream
    xs.thisEntity = jid.JID('thisHost')
    xs.otherEntity = jid.JID('otherHost')
    xs.initiating = False
    xs.sid = 'session01'
    xs.sendHeader()
    splitHeader = xs.transport.value()[0:-1].split(b' ')
    self.assertIn(b"to='otherhost'", splitHeader)
    self.assertIn(b"from='thishost'", splitHeader)
    self.assertIn(b"id='session01'", splitHeader)