import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_startProcessUnknownKeyError(self):
    """
        L{ProcessMonitor.startProcess} raises a C{KeyError} if the given
        process name isn't recognised.
        """
    self.assertRaises(KeyError, self.pm.startProcess, 'foo')