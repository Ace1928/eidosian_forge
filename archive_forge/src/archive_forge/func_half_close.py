import base64
import errno
import json
from multiprocessing import connection
from multiprocessing import managers
import socket
import struct
import weakref
from oslo_rootwrap import wrapper
def half_close(self):
    self._socket.shutdown(socket.SHUT_RD)