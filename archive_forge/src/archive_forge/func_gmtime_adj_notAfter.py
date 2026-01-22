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
def gmtime_adj_notAfter(self, amount: int) -> None:
    """
        Adjust the time stamp on which the certificate stops being valid.

        :param int amount: The number of seconds by which to adjust the
            timestamp.
        :return: ``None``
        """
    if not isinstance(amount, int):
        raise TypeError('amount must be an integer')
    notAfter = _lib.X509_getm_notAfter(self._x509)
    _lib.X509_gmtime_adj(notAfter, amount)