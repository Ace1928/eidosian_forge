import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, TextIO, TypeVar
from unittest import TestCase
from zope.interface import implementer
from attrs import frozen
from typing_extensions import Protocol, TypedDict
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.error import ProcessDone
from twisted.internet.interfaces import IAddress, ITransport
from twisted.internet.protocol import ProcessProtocol
from twisted.logger import Logger
from twisted.protocols.amp import AMP
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedObject
from twisted.trial._dist import (
from twisted.trial._dist.workerreporter import WorkerReporter
from twisted.trial.reporter import TestResult
from twisted.trial.runner import TestLoader, TrialSuite
from twisted.trial.unittest import Todo
from .stream import StreamOpen, StreamReceiver, StreamWrite
def connectionMade(self):
    """
        When connection is made, create the AMP protocol instance.
        """
    self._ampProtocol.makeConnection(LocalWorkerTransport(self.transport))
    self._logDirectory.makedirs(ignoreExistingDirectory=True)
    self._outLog = self._logDirectory.child('out.log').open('w')
    self._errLog = self._logDirectory.child('err.log').open('w')
    self._ampProtocol.setTestStream(self._logFile)
    d = self._ampProtocol.callRemote(workercommands.Start, directory=self._logDirectory.path)
    d.addErrback(lambda x: None)