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
def whenDone(self) -> Deferred[Iterator[_TaskResultT]]:
    """
        Get a L{Deferred} notification of when this task is complete.

        @return: a L{Deferred} that fires with the C{iterator} that this
            L{CooperativeTask} was created with when the iterator has been
            exhausted (i.e. its C{next} method has raised C{StopIteration}), or
            fails with the exception raised by C{next} if it raises some other
            exception.

        @rtype: L{Deferred}
        """
    d: Deferred[Iterator[_TaskResultT]] = Deferred()
    if self._completionState is None:
        self._deferreds.append(d)
    else:
        assert self._completionResult is not None
        d.callback(self._completionResult)
    return d