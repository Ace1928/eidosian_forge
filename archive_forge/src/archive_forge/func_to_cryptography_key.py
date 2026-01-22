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
def to_cryptography_key(self) -> _Key:
    """
        Export as a ``cryptography`` key.

        :rtype: One of ``cryptography``'s `key interfaces`_.

        .. _key interfaces: https://cryptography.io/en/latest/hazmat/            primitives/asymmetric/rsa/#key-interfaces

        .. versionadded:: 16.1.0
        """
    from cryptography.hazmat.primitives.serialization import load_der_private_key, load_der_public_key
    if self._only_public:
        der = dump_publickey(FILETYPE_ASN1, self)
        return load_der_public_key(der)
    else:
        der = dump_privatekey(FILETYPE_ASN1, self)
        return load_der_private_key(der, None)