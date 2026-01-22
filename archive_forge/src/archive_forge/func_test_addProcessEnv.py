import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_addProcessEnv(self):
    """
        L{ProcessMonitor.addProcess} takes an C{env} parameter that is passed to
        L{IReactorProcess.spawnProcess}.
        """
    fakeEnv = {'KEY': 'value'}
    self.pm.startService()
    self.pm.addProcess('foo', ['foo'], uid=1, gid=2, env=fakeEnv)
    self.reactor.advance(0)
    self.assertEqual(self.reactor.spawnedProcesses[0]._environment, fakeEnv)