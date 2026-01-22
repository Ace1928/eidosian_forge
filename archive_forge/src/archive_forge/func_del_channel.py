import select
import socket
import sys
import time
import warnings
import os
from errno import EALREADY, EINPROGRESS, EWOULDBLOCK, ECONNRESET, EINVAL, \
def del_channel(self, map=None):
    fd = self._fileno
    if map is None:
        map = self._map
    if fd in map:
        del map[fd]
    self._fileno = None