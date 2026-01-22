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
def load_crl(type: int, buffer: Union[str, bytes]) -> CRL:
    """
    Load Certificate Revocation List (CRL) data from a string *buffer*.
    *buffer* encoded with the type *type*.

    :param type: The file type (one of FILETYPE_PEM, FILETYPE_ASN1)
    :param buffer: The buffer the CRL is stored in

    :return: The CRL object
    """
    if isinstance(buffer, str):
        buffer = buffer.encode('ascii')
    bio = _new_mem_buf(buffer)
    if type == FILETYPE_PEM:
        crl = _lib.PEM_read_bio_X509_CRL(bio, _ffi.NULL, _ffi.NULL, _ffi.NULL)
    elif type == FILETYPE_ASN1:
        crl = _lib.d2i_X509_CRL_bio(bio, _ffi.NULL)
    else:
        raise ValueError('type argument must be FILETYPE_PEM or FILETYPE_ASN1')
    if crl == _ffi.NULL:
        _raise_current_error()
    result = CRL.__new__(CRL)
    result._crl = _ffi.gc(crl, _lib.X509_CRL_free)
    return result