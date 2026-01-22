import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def total_renegotiations(self):
    """
        Find out the total number of renegotiations.

        :return: The number of renegotiations.
        :rtype: int
        """
    return _lib.SSL_total_renegotiations(self._ssl)