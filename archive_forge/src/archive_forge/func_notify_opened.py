import importlib
import inspect
import os
import warnings
from eventlet import patcher
from eventlet.support import greenlets as greenlet
from eventlet import timeout
def notify_opened(fd):
    """
    Some file descriptors may be closed 'silently' - that is, by the garbage
    collector, by an external library, etc. When the OS returns a file descriptor
    from an open call (or something similar), this may be the only indication we
    have that the FD has been closed and then recycled.
    We let the hub know that the old file descriptor is dead; any stuck listeners
    will be disabled and notified in turn.
    """
    hub = get_hub()
    hub.mark_as_reopened(fd)