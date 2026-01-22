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
def test_receiveTwoNetstrings(self):
    """
        A stream of two netstrings can be received in two portions,
        where the first portion contains the complete first netstring
        and the length specification of the second netstring.
        """
    self.netstringReceiver.dataReceived(b'1:a,1')
    self.assertTrue(self.netstringReceiver._payloadComplete())
    self.assertEqual(self.netstringReceiver.received, [b'a'])
    self.netstringReceiver.dataReceived(b':b,')
    self.assertEqual(self.netstringReceiver.received, [b'a', b'b'])