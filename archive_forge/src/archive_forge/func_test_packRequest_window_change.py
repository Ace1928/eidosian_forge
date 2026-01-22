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
def test_packRequest_window_change(self):
    """
        See test_parseRequest_window_change for the payload format.
        """
    self.assertEqual(session.packRequest_window_change((2, 1, 3, 4)), struct.pack('>4L', 1, 2, 3, 4))