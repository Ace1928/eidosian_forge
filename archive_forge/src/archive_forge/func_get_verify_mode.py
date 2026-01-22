import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_verify_mode(self):
    """
        Retrieve the Connection object's verify mode, as set by
        :meth:`set_verify`.

        :return: The verify mode
        """
    return _lib.SSL_get_verify_mode(self._ssl)