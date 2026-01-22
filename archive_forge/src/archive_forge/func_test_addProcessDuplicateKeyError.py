import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
def test_addProcessDuplicateKeyError(self):
    """
        L{ProcessMonitor.addProcess} raises a C{KeyError} if a process with the
        given name already exists.
        """
    self.pm.addProcess('foo', ['arg1', 'arg2'], uid=1, gid=2, env={})
    self.assertRaises(KeyError, self.pm.addProcess, 'foo', ['arg1', 'arg2'], uid=1, gid=2, env={})