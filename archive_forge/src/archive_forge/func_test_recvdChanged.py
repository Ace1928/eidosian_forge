import struct
import sys
from io import BytesIO
from typing import List, Optional, Type
from zope.interface.verify import verifyObject
from twisted.internet import protocol, task
from twisted.internet.interfaces import IProducer
from twisted.internet.protocol import connectionDone
from twisted.protocols import basic
from twisted.python.compat import iterbytes
from twisted.python.failure import Failure
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_recvdChanged(self):
    """
        In stringReceived, if recvd is changed, messages should be parsed from
        it rather than the input to dataReceived.
        """
    r = self.getProtocol()
    result = []
    payloadC = b'c' * 5
    messageC = self.makeMessage(r, payloadC)

    def stringReceived(receivedString):
        if not result:
            r.recvd = messageC
        result.append(receivedString)
    r.stringReceived = stringReceived
    payloadA = b'a' * 5
    payloadB = b'b' * 5
    messageA = self.makeMessage(r, payloadA)
    messageB = self.makeMessage(r, payloadB)
    r.dataReceived(messageA + messageB)
    self.assertEqual(result, [payloadA, payloadC])