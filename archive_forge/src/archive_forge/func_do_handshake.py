import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def do_handshake(self):
    """
        Perform an SSL handshake (usually called after :meth:`renegotiate` or
        one of :meth:`set_accept_state` or :meth:`set_connect_state`). This can
        raise the same exceptions as :meth:`send` and :meth:`recv`.

        :return: None.
        """
    result = _lib.SSL_do_handshake(self._ssl)
    self._raise_ssl_error(self._ssl, result)