import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_reprLooksGood(self):
    """
        Repr includes all details
        """
    self.pm.addProcess('foo', ['arg1', 'arg2'], uid=1, gid=2, env={})
    representation = repr(self.pm)
    self.assertIn('foo', representation)
    self.assertIn('1', representation)
    self.assertIn('2', representation)