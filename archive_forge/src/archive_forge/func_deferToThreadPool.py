from __future__ import annotations
import queue as Queue
from typing import Callable, TypeVar
from typing_extensions import ParamSpec
from twisted.internet import defer
from twisted.internet.interfaces import IReactorFromThreads
from twisted.python import failure
from twisted.python.threadpool import ThreadPool
def deferToThreadPool(reactor: IReactorFromThreads, threadpool: ThreadPool, f: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> defer.Deferred[_R]:
    """
    Call the function C{f} using a thread from the given threadpool and return
    the result as a Deferred.

    This function is only used by client code which is maintaining its own
    threadpool.  To run a function in the reactor's threadpool, use
    C{deferToThread}.

    @param reactor: The reactor in whose main thread the Deferred will be
        invoked.

    @param threadpool: An object which supports the C{callInThreadWithCallback}
        method of C{twisted.python.threadpool.ThreadPool}.

    @param f: The function to call.
    @param args: positional arguments to pass to f.
    @param kwargs: keyword arguments to pass to f.

    @return: A Deferred which fires a callback with the result of f, or an
        errback with a L{twisted.python.failure.Failure} if f throws an
        exception.
    """
    d: defer.Deferred[_R] = defer.Deferred()

    def onResult(success: bool, result: _R | BaseException) -> None:
        if success:
            reactor.callFromThread(d.callback, result)
        else:
            reactor.callFromThread(d.errback, result)
    threadpool.callInThreadWithCallback(onResult, f, *args, **kwargs)
    return d