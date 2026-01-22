import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_simpleReprLooksGood(self):
    """
        Repr does not include unneeded details.

        Values of attributes that just mean "inherit from launching
        process" do not appear in the repr of a process.
        """
    self.pm.addProcess('foo', ['arg1', 'arg2'], env={})
    representation = repr(self.pm)
    self.assertNotIn('(', representation)
    self.assertNotIn(')', representation)