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
def get_signature_algorithm(self) -> bytes:
    """
        Return the signature algorithm used in the certificate.

        :return: The name of the algorithm.
        :rtype: :py:class:`bytes`

        :raises ValueError: If the signature algorithm is undefined.

        .. versionadded:: 0.13
        """
    algor = _lib.X509_get0_tbs_sigalg(self._x509)
    nid = _lib.OBJ_obj2nid(algor.algorithm)
    if nid == _lib.NID_undef:
        raise ValueError('Undefined signature algorithm')
    return _ffi.string(_lib.OBJ_nid2ln(nid))