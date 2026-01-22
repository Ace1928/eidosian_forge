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
def test_eofReceived(self):
    """
        When an EOF is received and an ISession adapter is present, it should
        be notified of the EOF message.
        """
    self.session.session = session.ISession(self.session.avatar)
    self.session.eofReceived()
    self.assertTrue(self.session.session.gotEOF)