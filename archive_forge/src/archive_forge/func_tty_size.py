import errno
import fcntl
import os
from oslo_log import log as logging
import select
import signal
import socket
import ssl
import struct
import sys
import termios
import time
import tty
from urllib import parse as urlparse
import websocket
from zunclient.common.apiclient import exceptions as acexceptions
from zunclient.common.websocketclient import exceptions
def tty_size(self, fd):
    """Get the tty size

        Return a tuple (rows,cols) representing the size of the TTY `fd`.

        The provided file descriptor should be the stdout stream of the TTY.

        If the TTY size cannot be determined, returns None.
        """
    if not os.isatty(fd.fileno()):
        return None
    try:
        dims = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, 'hhhh'))
    except Exception:
        try:
            dims = (os.environ['LINES'], os.environ['COLUMNS'])
        except Exception:
            return None
    return dims