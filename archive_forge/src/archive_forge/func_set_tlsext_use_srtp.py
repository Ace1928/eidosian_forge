import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
def set_tlsext_use_srtp(self, profiles):
    """
        Enable support for negotiating SRTP keying material.

        :param bytes profiles: A colon delimited list of protection profile
            names, like ``b'SRTP_AES128_CM_SHA1_80:SRTP_AES128_CM_SHA1_32'``.
        :return: None
        """
    if not isinstance(profiles, bytes):
        raise TypeError('profiles must be a byte string.')
    _openssl_assert(_lib.SSL_CTX_set_tlsext_use_srtp(self._context, profiles) == 0)