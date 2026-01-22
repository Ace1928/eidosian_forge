import select
import socket
import sys
import time
import warnings
import os
from errno import EALREADY, EINPROGRESS, EWOULDBLOCK, ECONNRESET, EINVAL, \
def handle_expt_event(self):
    err = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
    if err != 0:
        self.handle_close()
    else:
        self.handle_expt()