import select
import socket
import sys
import time
import warnings
import os
from errno import EALREADY, EINPROGRESS, EWOULDBLOCK, ECONNRESET, EINVAL, \
def create_socket(self, family=socket.AF_INET, type=socket.SOCK_STREAM):
    self.family_and_type = (family, type)
    sock = socket.socket(family, type)
    sock.setblocking(False)
    self.set_socket(sock)