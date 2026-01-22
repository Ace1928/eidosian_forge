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
def test_notQuiteMaximumLineLengthUnfinished(self):
    """
        C{LineReceiver} doesn't disconnect the transport it if
        receives a non-finished line whose length, counting the
        delimiter, is longer than its C{MAX_LENGTH} but shorter than
        its C{MAX_LENGTH} + len(delimiter). (When the first part that
        exceeds the max is the beginning of the delimiter.)
        """
    proto = basic.LineReceiver()
    proto.delimiter = b'\r\n'
    transport = proto_helpers.StringTransport()
    proto.makeConnection(transport)
    proto.dataReceived(b'x' * proto.MAX_LENGTH + proto.delimiter[:len(proto.delimiter) - 1])
    self.assertFalse(transport.disconnecting)