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
def get_serial_number(self) -> int:
    """
        Return the serial number of this certificate.

        :return: The serial number.
        :rtype: int
        """
    asn1_serial = _lib.X509_get_serialNumber(self._x509)
    bignum_serial = _lib.ASN1_INTEGER_to_BN(asn1_serial, _ffi.NULL)
    try:
        hex_serial = _lib.BN_bn2hex(bignum_serial)
        try:
            hexstring_serial = _ffi.string(hex_serial)
            serial = int(hexstring_serial, 16)
            return serial
        finally:
            _lib.OPENSSL_free(hex_serial)
    finally:
        _lib.BN_free(bignum_serial)