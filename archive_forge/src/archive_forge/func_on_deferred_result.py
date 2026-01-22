from typing import List, TYPE_CHECKING
from functools import partial
from testtools.content import TracebackContent
def on_deferred_result(deferred, on_success, on_failure, on_no_result):
    """Handle the result of a synchronous ``Deferred``.

    If ``deferred`` has fire successfully, call ``on_success``.
    If ``deferred`` has failed, call ``on_failure``.
    If ``deferred`` has not yet fired, call ``on_no_result``.

    The value of ``deferred`` will be preserved, so that other callbacks and
    errbacks can be added to ``deferred``.

    :param Deferred[A] deferred: A synchronous Deferred.
    :param Callable[[Deferred[A], A], T] on_success: Called if the Deferred
        fires successfully.
    :param Callable[[Deferred[A], Failure], T] on_failure: Called if the
        Deferred fires unsuccessfully.
    :param Callable[[Deferred[A]], T] on_no_result: Called if the Deferred has
        not yet fired.

    :raises ImpossibleDeferredError: If the Deferred somehow
        triggers both a success and a failure.
    :raises TypeError: If the Deferred somehow triggers more than one success,
        or more than one failure.

    :return: Whatever is returned by the triggered callback.
    :rtype: ``T``
    """
    successes: List['Deferred'] = []
    failures: List['Failure'] = []

    def capture(value, values):
        values.append(value)
        return value
    deferred.addCallbacks(partial(capture, values=successes), partial(capture, values=failures))
    if successes and failures:
        raise ImpossibleDeferredError(deferred, successes, failures)
    elif failures:
        [failure] = failures
        return on_failure(deferred, failure)
    elif successes:
        [result] = successes
        return on_success(deferred, result)
    else:
        return on_no_result(deferred)