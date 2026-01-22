import base64
import errno
import json
from multiprocessing import connection
from multiprocessing import managers
import socket
import struct
import weakref
from oslo_rootwrap import wrapper
def get_accepted(self):
    return self._accepted