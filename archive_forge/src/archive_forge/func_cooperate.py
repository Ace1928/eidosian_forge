import sys
import time
import warnings
from typing import (
from zope.interface import implementer
from incremental import Version
from twisted.internet.base import DelayedCall
from twisted.internet.defer import Deferred, ensureDeferred, maybeDeferred
from twisted.internet.error import ReactorNotRunning
from twisted.internet.interfaces import IDelayedCall, IReactorCore, IReactorTime
from twisted.python import log, reflect
from twisted.python.deprecate import _getDeprecationWarningString
from twisted.python.failure import Failure
def cooperate(self, iterator: Iterator[_TaskResultT]) -> CooperativeTask:
    """
        Start running the given iterator as a long-running cooperative task, by
        calling next() on it as a periodic timed event.

        @param iterator: the iterator to invoke.

        @return: a L{CooperativeTask} object representing this task.
        """
    return CooperativeTask(iterator, self)