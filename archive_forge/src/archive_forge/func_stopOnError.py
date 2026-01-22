import os
import signal
import time
from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Type, Union, cast
from zope.interface import Interface
from twisted.python import log
from twisted.python.deprecate import _fullyQualifiedName as fullyQualifiedName
from twisted.python.failure import Failure
from twisted.python.reflect import namedAny
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, SynchronousTestCase
from twisted.trial.util import DEFAULT_TIMEOUT_DURATION, acquireAttribute
def stopOnError(case, reactor, publisher=None):
    """
    Stop the reactor as soon as any error is logged on the given publisher.

    This is beneficial for tests which will wait for a L{Deferred} to fire
    before completing (by passing or failing).  Certain implementation bugs may
    prevent the L{Deferred} from firing with any result at all (consider a
    protocol's {dataReceived} method that raises an exception: this exception
    is logged but it won't ever cause a L{Deferred} to fire).  In that case the
    test would have to complete by timing out which is a much less desirable
    outcome than completing as soon as the unexpected error is encountered.

    @param case: A L{SynchronousTestCase} to use to clean up the necessary log
        observer when the test is over.
    @param reactor: The reactor to stop.
    @param publisher: A L{LogPublisher} to watch for errors.  If L{None}, the
        global log publisher will be watched.
    """
    if publisher is None:
        from twisted.python import log as publisher
    running = [None]

    def stopIfError(event):
        if running and event.get('isError'):
            running.pop()
            reactor.stop()
    publisher.addObserver(stopIfError)
    case.addCleanup(publisher.removeObserver, stopIfError)