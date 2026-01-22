import os
import json
import atexit
import abc
import enum
import time
import threading
from timeit import default_timer as timer
from contextlib import contextmanager, ExitStack
from collections import defaultdict
from numba.core import config
@contextmanager
def install_timer(kind, callback):
    """Install a TimingListener temporarily to measure the duration of
    an event.

    If the context completes successfully, the *callback* function is executed.
    The *callback* function is expected to take a float argument for the
    duration in seconds.

    Returns
    -------
    res : TimingListener

    Examples
    --------

    This is equivalent to:

    >>> with install_listener(kind, TimingListener()) as res:
    >>>    ...
    """
    tl = TimingListener()
    with install_listener(kind, tl):
        yield tl
    if tl.done:
        callback(tl.duration)