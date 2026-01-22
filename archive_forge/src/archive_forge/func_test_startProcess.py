import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_startProcess(self):
    """
        When a process has been started, an instance of L{LoggingProtocol} will
        be added to the L{ProcessMonitor.protocols} dict and the start time of
        the process will be recorded in the L{ProcessMonitor.timeStarted}
        dictionary.
        """
    self.pm.addProcess('foo', ['foo'])
    self.pm.startProcess('foo')
    self.assertIsInstance(self.pm.protocols['foo'], LoggingProtocol)
    self.assertIn('foo', self.pm.timeStarted.keys())