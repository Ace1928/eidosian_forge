from __future__ import print_function
import os
import socket
import signal
import threading
from contextlib import closing, contextmanager
from . import _gi
def sigint_handler_is_default():
    """Returns if on SIGINT the default Python handler would be called"""
    return signal.getsignal(signal.SIGINT) is signal.default_int_handler and PyOS_getsig(signal.SIGINT) == startup_sigint_ptr