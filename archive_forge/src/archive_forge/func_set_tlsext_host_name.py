import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_tlsext_host_name(self, name):
    """
        Set the value of the servername extension to send in the client hello.

        :param name: A byte string giving the name.

        .. versionadded:: 0.13
        """
    if not isinstance(name, bytes):
        raise TypeError('name must be a byte string')
    elif b'\x00' in name:
        raise TypeError('name must not contain NUL byte')
    _lib.SSL_set_tlsext_host_name(self._ssl, name)