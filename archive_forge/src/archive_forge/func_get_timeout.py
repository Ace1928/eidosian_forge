import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_timeout(self):
    """
        Retrieve session timeout, as set by :meth:`set_timeout`. The default
        is 300 seconds.

        :return: The session timeout
        """
    return _lib.SSL_CTX_get_timeout(self._context)