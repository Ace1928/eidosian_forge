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
def test_excessiveKeyFailure(self):
    """
        If L{amp.BinaryBoxProtocol} disconnects because it received a key
        length prefix which was too large, the L{IBoxReceiver}'s
        C{stopReceivingBoxes} method is called with a L{TooLong} failure.
        """
    protocol = amp.BinaryBoxProtocol(self)
    protocol.makeConnection(StringTransport())
    protocol.dataReceived(b'\x01\x00')
    protocol.connectionLost(Failure(error.ConnectionDone('simulated connection done')))
    self.stopReason.trap(amp.TooLong)
    self.assertTrue(self.stopReason.value.isKey)
    self.assertFalse(self.stopReason.value.isLocal)
    self.assertIsNone(self.stopReason.value.value)
    self.assertIsNone(self.stopReason.value.keyName)