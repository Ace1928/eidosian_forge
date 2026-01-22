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
@skipIf(not hasattr(os, 'WCOREDUMP'), "can't run this w/o os.WCOREDUMP")
def test_processEndedWithExitSignalNoCoreDump(self):
    """
        When processEnded is called, if there is an exit signal in the
        reason it should be sent in an exit-signal message.  If no
        core was dumped, don't set the core-dump bit.
        """
    self.pp.processEnded(Failure(ProcessTerminated(1, signal.SIGTERM, 0)))
    self.assertRequestsEqual([(b'exit-signal', common.NS(b'TERM') + b'\x00' + common.NS(b'') + common.NS(b''), False)])
    self.assertSessionClosed()