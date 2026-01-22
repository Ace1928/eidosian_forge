import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def want_write(self):
    """
        Checks if there is data to write to the transport layer to complete an
        operation.

        :return: True iff there is data to write
        """
    return _lib.SSL_want_write(self._ssl)