import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_connectionLostMurderCancel(self):
    """
        L{ProcessMonitor.connectionLost} cancels a scheduled process killer and
        deletes the DelayedCall from the L{ProcessMonitor.murder} list.
        """
    self.pm.addProcess('foo', ['foo'])
    self.pm.startService()
    self.reactor.advance(1)
    self.pm.stopProcess('foo')
    self.assertIn('foo', self.pm.murder)
    delayedCall = self.pm.murder['foo']
    self.assertTrue(delayedCall.active())
    self.reactor.advance(self.pm.protocols['foo'].transport._terminationDelay)
    self.assertFalse(delayedCall.active())
    self.assertNotIn('foo', self.pm.murder)