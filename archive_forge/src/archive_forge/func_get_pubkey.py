import calendar
import datetime
import functools
from base64 import b16encode
from functools import partial
from os import PathLike
from typing import (
from cryptography import utils, x509
from cryptography.hazmat.primitives.asymmetric import (
from OpenSSL._util import (
def get_pubkey(self) -> PKey:
    """
        Get the public key of this certificate.

        :return: The public key.
        :rtype: :py:class:`PKey`
        """
    pkey = PKey.__new__(PKey)
    pkey._pkey = _lib.NETSCAPE_SPKI_get_pubkey(self._spki)
    _openssl_assert(pkey._pkey != _ffi.NULL)
    pkey._pkey = _ffi.gc(pkey._pkey, _lib.EVP_PKEY_free)
    pkey._only_public = True
    return pkey