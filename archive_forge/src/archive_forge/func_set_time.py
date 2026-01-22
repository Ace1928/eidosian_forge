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
def set_time(self, vfy_time: datetime.datetime) -> None:
    """
        Set the time against which the certificates are verified.

        Normally the current time is used.

        .. note::

          For example, you can determine if a certificate was valid at a given
          time.

        .. versionadded:: 17.0.0

        :param datetime vfy_time: The verification time to set on this store.
        :return: ``None`` if the verification time was successfully set.
        """
    param = _lib.X509_VERIFY_PARAM_new()
    param = _ffi.gc(param, _lib.X509_VERIFY_PARAM_free)
    _lib.X509_VERIFY_PARAM_set_time(param, calendar.timegm(vfy_time.timetuple()))
    _openssl_assert(_lib.X509_STORE_set1_param(self._store, param) != 0)