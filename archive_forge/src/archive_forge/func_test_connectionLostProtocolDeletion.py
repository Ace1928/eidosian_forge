import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_connectionLostProtocolDeletion(self):
    """
        L{ProcessMonitor.connectionLost} removes the corresponding
        ProcessProtocol instance from the L{ProcessMonitor.protocols} list.
        """
    self.pm.startService()
    self.pm.addProcess('foo', ['foo'])
    self.assertIn('foo', self.pm.protocols)
    self.pm.protocols['foo'].transport.signalProcess('KILL')
    self.reactor.advance(self.pm.protocols['foo'].transport._terminationDelay)
    self.assertNotIn('foo', self.pm.protocols)