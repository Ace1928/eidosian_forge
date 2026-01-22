import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_state_string(self):
    """
        Retrieve a verbose string detailing the state of the Connection.

        :return: A string representing the state
        :rtype: bytes
        """
    return _ffi.string(_lib.SSL_state_string_long(self._ssl))