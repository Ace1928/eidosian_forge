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
def test_receiveNetstringPortions_3(self):
    """
        Netstrings can be received one character at a time.
        """
    for part in [b'2', b':', b'a', b'b', b',']:
        self.netstringReceiver.dataReceived(part)
    self.assertEqual(self.netstringReceiver.received, [b'ab'])