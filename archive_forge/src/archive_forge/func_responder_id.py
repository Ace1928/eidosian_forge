from __future__ import annotations
import abc
import datetime
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import ocsp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.base import (
def responder_id(self, encoding: OCSPResponderEncoding, responder_cert: x509.Certificate) -> OCSPResponseBuilder:
    if self._responder_id is not None:
        raise ValueError('responder_id can only be set once')
    if not isinstance(responder_cert, x509.Certificate):
        raise TypeError('responder_cert must be a Certificate')
    if not isinstance(encoding, OCSPResponderEncoding):
        raise TypeError('encoding must be an element from OCSPResponderEncoding')
    return OCSPResponseBuilder(self._response, (responder_cert, encoding), self._certs, self._extensions)