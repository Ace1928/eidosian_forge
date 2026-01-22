import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def export_keying_material(self, label, olen, context=None):
    """
        Obtain keying material for application use.

        :param: label - a disambiguating label string as described in RFC 5705
        :param: olen - the length of the exported key material in bytes
        :param: context - a per-association context value
        :return: the exported key material bytes or None
        """
    outp = _no_zero_allocator('unsigned char[]', olen)
    context_buf = _ffi.NULL
    context_len = 0
    use_context = 0
    if context is not None:
        context_buf = context
        context_len = len(context)
        use_context = 1
    success = _lib.SSL_export_keying_material(self._ssl, outp, olen, label, len(label), context_buf, context_len, use_context)
    _openssl_assert(success == 1)
    return _ffi.buffer(outp, olen)[:]