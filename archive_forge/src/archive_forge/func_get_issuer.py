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
def get_issuer(self) -> X509Name:
    """
        Get the CRL's issuer.

        .. versionadded:: 16.1.0

        :rtype: X509Name
        """
    _issuer = _lib.X509_NAME_dup(_lib.X509_CRL_get_issuer(self._crl))
    _openssl_assert(_issuer != _ffi.NULL)
    _issuer = _ffi.gc(_issuer, _lib.X509_NAME_free)
    issuer = X509Name.__new__(X509Name)
    issuer._name = _issuer
    return issuer