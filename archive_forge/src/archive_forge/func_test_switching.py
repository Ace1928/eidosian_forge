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
def test_switching(self):
    """
        Data already parsed by L{IntNStringReceiver.dataReceived} is not
        reparsed if C{stringReceived} consumes some of the
        L{IntNStringReceiver.recvd} buffer.
        """
    proto = self.getProtocol()
    mix = []
    SWITCH = b'\x00\x00\x00\x00'
    for s in self.strings:
        mix.append(self.makeMessage(proto, s))
        mix.append(SWITCH)
    result = []

    def stringReceived(receivedString):
        result.append(receivedString)
        proto.recvd = proto.recvd[len(SWITCH):]
    proto.stringReceived = stringReceived
    proto.dataReceived(b''.join(mix))
    proto.dataReceived(b'\x01')
    self.assertEqual(result, self.strings)
    self.assertEqual(proto.recvd, b'\x01')