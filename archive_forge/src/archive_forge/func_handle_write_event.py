import select
import socket
import sys
import time
import warnings
import os
from errno import EALREADY, EINPROGRESS, EWOULDBLOCK, ECONNRESET, EINVAL, \
def handle_write_event(self):
    if self.accepting:
        return
    if not self.connected:
        if self.connecting:
            self.handle_connect_event()
    self.handle_write()