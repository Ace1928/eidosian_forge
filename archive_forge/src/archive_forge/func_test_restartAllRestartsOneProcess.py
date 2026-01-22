import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_restartAllRestartsOneProcess(self):
    """
        L{ProcessMonitor.restartAll} succeeds when there is one process.
        """
    self.pm.addProcess('foo', ['foo'])
    self.pm.startService()
    self.reactor.advance(1)
    self.pm.restartAll()
    self.reactor.advance(1)
    processes = list(self.reactor.spawnedProcesses)
    myProcess = processes.pop()
    self.assertEquals(processes, [])
    self.assertIsNone(myProcess.pid)