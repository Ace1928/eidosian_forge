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
def test_abortedTransfer(self):
    """
        The C{Deferred} returned by L{basic.FileSender.beginFileTransfer} fails
        with an C{Exception} if C{stopProducing} when the transfer is not
        complete.
        """
    source = BytesIO(b'Test content')
    consumer = proto_helpers.StringTransport()
    sender = basic.FileSender()
    d = sender.beginFileTransfer(source, consumer)
    sender.stopProducing()
    failure = self.failureResultOf(d)
    failure.trap(Exception)
    self.assertEqual('Consumer asked us to stop producing', str(failure.value))