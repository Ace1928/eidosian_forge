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
def verify_certificate(self) -> None:
    """
        Verify a certificate in a context.

        .. versionadded:: 0.15

        :raises X509StoreContextError: If an error occurred when validating a
          certificate in the context. Sets ``certificate`` attribute to
          indicate which certificate caused the error.
        """
    self._cleanup()
    self._init()
    ret = _lib.X509_verify_cert(self._store_ctx)
    self._cleanup()
    if ret <= 0:
        raise self._exception_from_context()