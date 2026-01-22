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
def revocation_date(self, time: datetime.datetime) -> RevokedCertificateBuilder:
    if not isinstance(time, datetime.datetime):
        raise TypeError('Expecting datetime object.')
    if self._revocation_date is not None:
        raise ValueError('The revocation date may only be set once.')
    time = _convert_to_naive_utc_time(time)
    if time < _EARLIEST_UTC_TIME:
        raise ValueError('The revocation date must be on or after 1950 January 1.')
    return RevokedCertificateBuilder(self._serial_number, time, self._extensions)