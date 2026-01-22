import os
import signal
import struct
import sys
from unittest import skipIf
from zope.interface import implementer
from twisted.internet import defer, error, protocol
from twisted.internet.address import IPv4Address
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.python import components, failure
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.python.test.test_components import RegistryUsingMixin
from twisted.trial.unittest import TestCase
def test_parseRequest_window_change(self):
    """
        The payload of a window_change request is::
            uint32  columns
            uint32  rows
            uint32  x pixels
            uint32  y pixels

        parseRequest_window_change() returns (rows, columns, x pixels,
        y pixels).
        """
    self.assertEqual(session.parseRequest_window_change(struct.pack('>4L', 1, 2, 3, 4)), (2, 1, 3, 4))