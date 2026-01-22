from __future__ import absolute_import
import errno
import logging
import select
import socket
import time
from serial.serialutil import SerialBase, SerialException, to_bytes, \
@property
def in_waiting(self):
    """Return the number of bytes currently in the input buffer."""
    if not self.is_open:
        raise PortNotOpenError()
    lr, lw, lx = select.select([self._socket], [], [], 0)
    return len(lr)