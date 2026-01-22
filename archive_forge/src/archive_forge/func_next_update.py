from __future__ import annotations
import abc
import datetime
import os
import typing
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.extensions import (
from cryptography.x509.name import Name, _ASN1Type
from cryptography.x509.oid import ObjectIdentifier
def next_update(self, next_update: datetime.datetime) -> CertificateRevocationListBuilder:
    if not isinstance(next_update, datetime.datetime):
        raise TypeError('Expecting datetime object.')
    if self._next_update is not None:
        raise ValueError('Last update may only be set once.')
    next_update = _convert_to_naive_utc_time(next_update)
    if next_update < _EARLIEST_UTC_TIME:
        raise ValueError('The last update date must be on or after 1950 January 1.')
    if self._last_update is not None and next_update < self._last_update:
        raise ValueError('The next update date must be after the last update date.')
    return CertificateRevocationListBuilder(self._issuer_name, self._last_update, next_update, self._extensions, self._revoked_certificates)