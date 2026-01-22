import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def test_doubleProtocolSwitch(self):
    """
        As a debugging aid, a protocol system should raise a
        L{ProtocolSwitched} exception when asked to switch a protocol that is
        already switched.
        """
    serverDeferred = defer.Deferred()
    serverProto = SimpleSymmetricCommandProtocol(serverDeferred)
    clientDeferred = defer.Deferred()
    clientProto = SimpleSymmetricCommandProtocol(clientDeferred)
    c, s, p = connectedServerAndClient(ServerClass=lambda: serverProto, ClientClass=lambda: clientProto)

    def switched(result):
        self.assertRaises(amp.ProtocolSwitched, c.switchToTestProtocol)
        self.testSucceeded = True
    c.switchToTestProtocol().addCallback(switched)
    p.flush()
    self.assertTrue(self.testSucceeded)