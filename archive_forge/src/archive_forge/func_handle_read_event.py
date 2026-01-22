import select
import socket
import sys
import time
import warnings
import os
from errno import EALREADY, EINPROGRESS, EWOULDBLOCK, ECONNRESET, EINVAL, \
def handle_read_event(self):
    if self.accepting:
        self.handle_accept()
    elif not self.connected:
        if self.connecting:
            self.handle_connect_event()
        self.handle_read()
    else:
        self.handle_read()