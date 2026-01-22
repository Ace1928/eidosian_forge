import select
import socket
import sys
import time
import warnings
import os
from errno import EALREADY, EINPROGRESS, EWOULDBLOCK, ECONNRESET, EINVAL, \
def handle_accept(self):
    pair = self.accept()
    if pair is not None:
        self.handle_accepted(*pair)