from __future__ import print_function
import os
import socket
import signal
import threading
from contextlib import closing, contextmanager
from . import _gi
@contextmanager
def sigint_handler_set_and_restore_default(handler):
    """Context manager for saving/restoring the SIGINT handler default state.

    Will only restore the default handler again if the handler is not changed
    while the context is active.
    """
    assert sigint_handler_is_default()
    signal.signal(signal.SIGINT, handler)
    sig_ptr = PyOS_getsig(signal.SIGINT)
    try:
        yield
    finally:
        if signal.getsignal(signal.SIGINT) is handler and PyOS_getsig(signal.SIGINT) == sig_ptr:
            signal.signal(signal.SIGINT, signal.default_int_handler)