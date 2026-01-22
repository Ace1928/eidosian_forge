import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
@_requires_alpn
def set_alpn_select_callback(self, callback):
    """
        Specify a callback function that will be called on the server when a
        client offers protocols using ALPN.

        :param callback: The callback function.  It will be invoked with two
            arguments: the Connection, and a list of offered protocols as
            bytestrings, e.g ``[b'http/1.1', b'spdy/2']``.  It can return
            one of those bytestrings to indicate the chosen protocol, the
            empty bytestring to terminate the TLS connection, or the
            :py:obj:`NO_OVERLAPPING_PROTOCOLS` to indicate that no offered
            protocol was selected, but that the connection should not be
            aborted.
        """
    self._alpn_select_helper = _ALPNSelectHelper(callback)
    self._alpn_select_callback = self._alpn_select_helper.callback
    _lib.SSL_CTX_set_alpn_select_cb(self._context, self._alpn_select_callback, _ffi.NULL)