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
def set_alpn_protos(self, protos):
    """
        Specify the client's ALPN protocol list.

        These protocols are offered to the server during protocol negotiation.

        :param protos: A list of the protocols to be offered to the server.
            This list should be a Python list of bytestrings representing the
            protocols to offer, e.g. ``[b'http/1.1', b'spdy/2']``.
        """
    if not protos:
        raise ValueError('at least one protocol must be specified')
    protostr = b''.join(chain.from_iterable(((bytes((len(p),)), p) for p in protos)))
    input_str = _ffi.new('unsigned char[]', protostr)
    _openssl_assert(_lib.SSL_set_alpn_protos(self._ssl, input_str, len(protostr)) == 0)