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
def test_addHandler(self):
    """
        Test the addition of a protocol handler while not connected.
        """
    sm = self.streamManager
    handler = DummyXMPPHandler()
    handler.setHandlerParent(sm)
    self.assertEqual(0, handler.doneMade)
    self.assertEqual(0, handler.doneInitialized)
    self.assertEqual(0, handler.doneLost)