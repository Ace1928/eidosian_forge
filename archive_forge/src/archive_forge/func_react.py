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
def react(main: Callable[..., Union[Deferred[_T], Coroutine['Deferred[_T]', object, _T]]], argv: Iterable[object]=(), _reactor: Optional[IReactorCore]=None) -> NoReturn:
    """
    Call C{main} and run the reactor until the L{Deferred} it returns fires or
    the coroutine it returns completes.

    This is intended as the way to start up an application with a well-defined
    completion condition.  Use it to write clients or one-off asynchronous
    operations.  Prefer this to calling C{reactor.run} directly, as this
    function will also:

      - Take care to call C{reactor.stop} once and only once, and at the right
        time.
      - Log any failures from the C{Deferred} returned by C{main}.
      - Exit the application when done, with exit code 0 in case of success and
        1 in case of failure. If C{main} fails with a C{SystemExit} error, the
        code returned is used.

    The following demonstrates the signature of a C{main} function which can be
    used with L{react}::

      async def main(reactor, username, password):
          return "ok"

      task.react(main, ("alice", "secret"))

    @param main: A callable which returns a L{Deferred} or
        coroutine. It should take the reactor as its first
        parameter, followed by the elements of C{argv}.

    @param argv: A list of arguments to pass to C{main}. If omitted the
        callable will be invoked with no additional arguments.

    @param _reactor: An implementation detail to allow easier unit testing.  Do
        not supply this parameter.

    @since: 12.3
    """
    if _reactor is None:
        from twisted.internet import reactor
        _reactor = cast(IReactorCore, reactor)
    finished = ensureDeferred(main(_reactor, *argv))
    code = 0
    stopping = False

    def onShutdown() -> None:
        nonlocal stopping
        stopping = True
    _reactor.addSystemEventTrigger('before', 'shutdown', onShutdown)

    def stop(result: object, stopReactor: bool) -> None:
        if stopReactor:
            assert _reactor is not None
            try:
                _reactor.stop()
            except ReactorNotRunning:
                pass
        if isinstance(result, Failure):
            nonlocal code
            if result.check(SystemExit) is not None:
                code = result.value.code
            else:
                log.err(result, 'main function encountered error')
                code = 1

    def cbFinish(result: object) -> None:
        if stopping:
            stop(result, False)
        else:
            assert _reactor is not None
            _reactor.callWhenRunning(stop, result, True)
    finished.addBoth(cbFinish)
    _reactor.run()
    sys.exit(code)