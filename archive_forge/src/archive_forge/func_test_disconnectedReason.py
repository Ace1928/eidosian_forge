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
def test_disconnectedReason(self):
    """
        A L{STREAM_END_EVENT} results in L{StreamManager} firing the handlers
        L{connectionLost} methods, passing a L{failure.Failure} reason.
        """
    sm = self.streamManager
    handler = FailureReasonXMPPHandler()
    handler.setHandlerParent(sm)
    sm._disconnected(failure.Failure(Exception('no reason')))
    self.assertEqual(True, handler.gotFailureReason)