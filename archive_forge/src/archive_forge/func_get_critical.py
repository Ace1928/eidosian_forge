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
def get_critical(self) -> bool:
    """
        Returns the critical field of this X.509 extension.

        :return: The critical field.
        """
    return _lib.X509_EXTENSION_get_critical(self._extension)