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
def rawDataReceived(self, data):
    """
        Read raw data, until the quantity specified by a previous 'len' line is
        reached.
        """
    data, rest = (data[:self.length], data[self.length:])
    self.length = self.length - len(data)
    self.received[-1] = self.received[-1] + data
    if self.length == 0:
        self.setLineMode(rest)