import select
import socket
import sys
import time
import warnings
import os
from errno import EALREADY, EINPROGRESS, EWOULDBLOCK, ECONNRESET, EINVAL, \
def handle_connect_event(self):
    err = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
    if err != 0:
        raise OSError(err, _strerror(err))
    self.handle_connect()
    self.connected = True
    self.connecting = False