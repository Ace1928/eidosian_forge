import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def get_servername(self):
    """
        Retrieve the servername extension value if provided in the client hello
        message, or None if there wasn't one.

        :return: A byte string giving the server name or :data:`None`.

        .. versionadded:: 0.13
        """
    name = _lib.SSL_get_servername(self._ssl, _lib.TLSEXT_NAMETYPE_host_name)
    if name == _ffi.NULL:
        return None
    return _ffi.string(name)