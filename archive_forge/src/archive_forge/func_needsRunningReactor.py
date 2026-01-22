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
def needsRunningReactor(reactor, thunk):
    """
    Various functions within these tests need an already-running reactor at
    some point.  They need to stop the reactor when the test has completed, and
    that means calling reactor.stop().  However, reactor.stop() raises an
    exception if the reactor isn't already running, so if the L{Deferred} that
    a particular API under test returns fires synchronously (as especially an
    endpoint's C{connect()} method may do, if the connect is to a local
    interface address) then the test won't be able to stop the reactor being
    tested and finish.  So this calls C{thunk} only once C{reactor} is running.

    (This is just an alias for
    L{twisted.internet.interfaces.IReactorCore.callWhenRunning} on the given
    reactor parameter, in order to centrally reference the above paragraph and
    repeating it everywhere as a comment.)

    @param reactor: the L{twisted.internet.interfaces.IReactorCore} under test

    @param thunk: a 0-argument callable, which eventually finishes the test in
        question, probably in a L{Deferred} callback.
    """
    reactor.callWhenRunning(thunk)