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
def type_is_signed(self) -> bool:
    """
        Check if this NID_pkcs7_signed object

        :return: True if the PKCS7 is of type signed
        """
    return bool(_lib.PKCS7_type_is_signed(self._pkcs7))