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
def test_consumeLengthBorderCase2(self):
    """
        C{_consumeLength} raises a L{basic.NetstringParseError} if
        the length specification exceeds the value of C{MAX_LENGTH}
        by 1 (border case).
        """
    self.netstringReceiver._remainingData = b'12:'
    self.netstringReceiver.MAX_LENGTH = 11
    self.assertRaises(basic.NetstringParseError, self.netstringReceiver._consumeLength)