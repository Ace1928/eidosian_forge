import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def master_key(self):
    """
        Retrieve the value of the master key for this session.

        :return: A string representing the state
        """
    session = _lib.SSL_get_session(self._ssl)
    if session == _ffi.NULL:
        return None
    length = _lib.SSL_SESSION_get_master_key(session, _ffi.NULL, 0)
    _openssl_assert(length > 0)
    outp = _no_zero_allocator('unsigned char[]', length)
    _lib.SSL_SESSION_get_master_key(session, outp, length)
    return _ffi.buffer(outp, length)[:]