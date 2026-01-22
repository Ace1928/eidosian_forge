from __future__ import annotations
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
def exchange(self, algorithm: ec.ECDH, peer_public_key: ec.EllipticCurvePublicKey) -> bytes:
    if not self._backend.elliptic_curve_exchange_algorithm_supported(algorithm, self.curve):
        raise UnsupportedAlgorithm('This backend does not support the ECDH algorithm.', _Reasons.UNSUPPORTED_EXCHANGE_ALGORITHM)
    if peer_public_key.curve.name != self.curve.name:
        raise ValueError('peer_public_key and self are not on the same curve')
    return _evp_pkey_derive(self._backend, self._evp_pkey, peer_public_key)