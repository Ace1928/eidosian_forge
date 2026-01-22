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
def test_client_dataReceived(self):
    """
        SSHSession.dataReceived() passes data along to a client.  If the data
        comes before there is a client, the data should be discarded.
        """
    self.session.dataReceived(b'1')
    self.session.client = StubClient()
    self.session.dataReceived(b'2')
    self.assertEqual(self.session.client.transport.buf, b'2')