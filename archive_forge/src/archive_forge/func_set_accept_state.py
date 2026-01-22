import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_accept_state(self):
    """
        Set the connection to work in server mode. The handshake will be
        handled automatically by read/write.

        :return: None
        """
    _lib.SSL_set_accept_state(self._ssl)