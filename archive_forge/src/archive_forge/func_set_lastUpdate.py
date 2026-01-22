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
def set_lastUpdate(self, when: bytes) -> None:
    """
        Set when the CRL was last updated.

        The timestamp is formatted as an ASN.1 TIME::

            YYYYMMDDhhmmssZ

        .. versionadded:: 16.1.0

        :param bytes when: A timestamp string.
        :return: ``None``
        """
    lastUpdate = _new_asn1_time(when)
    ret = _lib.X509_CRL_set1_lastUpdate(self._crl, lastUpdate)
    _openssl_assert(ret == 1)