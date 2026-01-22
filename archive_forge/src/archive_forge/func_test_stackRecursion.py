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
def test_stackRecursion(self):
    """
        Test switching modes many times on the same data.
        """
    proto = FlippingLineTester()
    transport = proto_helpers.StringIOWithoutClosing()
    proto.makeConnection(protocol.FileWrapper(transport))
    limit = sys.getrecursionlimit()
    proto.dataReceived(b'x\nx' * limit)
    self.assertEqual(b'x' * limit, b''.join(proto.lines))