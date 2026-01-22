import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_connectionLostMinMaxRestartDelay(self):
    """
        L{ProcessMonitor.connectionLost} will wait at least minRestartDelay s
        and at most maxRestartDelay s
        """
    self.pm.minRestartDelay = 2
    self.pm.maxRestartDelay = 3
    self.pm.startService()
    self.pm.addProcess('foo', ['foo'])
    self.assertEqual(self.pm.delay['foo'], self.pm.minRestartDelay)
    self.reactor.advance(self.pm.threshold - 1)
    self.pm.protocols['foo'].processEnded(Failure(ProcessDone(0)))
    self.assertEqual(self.pm.delay['foo'], self.pm.maxRestartDelay)