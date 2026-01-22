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
def deferLater(clock: IReactorTime, delay: float, callable: Optional[Callable[..., _T]]=None, *args: object, **kw: object) -> Deferred[_T]:
    """
    Call the given function after a certain period of time has passed.

    @param clock: The object which will be used to schedule the delayed
        call.

    @param delay: The number of seconds to wait before calling the function.

    @param callable: The callable to call after the delay, or C{None}.

    @param args: The positional arguments to pass to C{callable}.

    @param kw: The keyword arguments to pass to C{callable}.

    @return: A deferred that fires with the result of the callable when the
        specified time has elapsed.
    """

    def deferLaterCancel(deferred: Deferred[object]) -> None:
        delayedCall.cancel()

    def cb(result: object) -> _T:
        if callable is None:
            return None
        return callable(*args, **kw)
    d: Deferred[_T] = Deferred(deferLaterCancel)
    d.addCallback(cb)
    delayedCall = clock.callLater(delay, d.callback, None)
    return d