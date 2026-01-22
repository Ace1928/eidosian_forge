import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_verify_depth(self, depth):
    """
        Set the maximum depth for the certificate chain verification that shall
        be allowed for this Context object.

        :param depth: An integer specifying the verify depth
        :return: None
        """
    if not isinstance(depth, int):
        raise TypeError('depth must be an integer')
    _lib.SSL_CTX_set_verify_depth(self._context, depth)