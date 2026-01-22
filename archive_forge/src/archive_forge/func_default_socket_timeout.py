import os
import socket
import sys
import threading
import traceback
from contextlib import contextmanager
from threading import TIMEOUT_MAX as THREAD_TIMEOUT_MAX
from celery.local import Proxy
@contextmanager
def default_socket_timeout(timeout):
    """Context temporarily setting the default socket timeout."""
    prev = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    yield
    socket.setdefaulttimeout(prev)