from collections import deque
import sys
from greenlet import GreenletExit
from eventlet import event
from eventlet import hubs
from eventlet import support
from eventlet import timeout
from eventlet.hubs import timer
from eventlet.support import greenlets as greenlet
import warnings
def spawn_n(func, *args, **kwargs):
    """Same as :func:`spawn`, but returns a ``greenlet`` object from
    which it is not possible to retrieve either a return value or
    whether it raised any exceptions.  This is faster than
    :func:`spawn`; it is fastest if there are no keyword arguments.

    If an exception is raised in the function, spawn_n prints a stack
    trace; the print can be disabled by calling
    :func:`eventlet.debug.hub_exceptions` with False.
    """
    return _spawn_n(0, func, args, kwargs)[1]