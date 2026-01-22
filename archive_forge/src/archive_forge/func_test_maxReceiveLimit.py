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
def test_maxReceiveLimit(self):
    """
        Netstrings with a length specification exceeding the specified
        C{MAX_LENGTH} are refused.
        """
    tooLong = self.netstringReceiver.MAX_LENGTH + 1
    self.netstringReceiver.dataReceived(b''.join((bytes(tooLong), b':', b'a' * tooLong)))
    self.assertTrue(self.transport.disconnecting)