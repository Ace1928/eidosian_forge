from __future__ import annotations
import abc
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization
@classmethod
def from_public_bytes(cls, data: bytes) -> X25519PublicKey:
    from cryptography.hazmat.backends.openssl.backend import backend
    if not backend.x25519_supported():
        raise UnsupportedAlgorithm('X25519 is not supported by this version of OpenSSL.', _Reasons.UNSUPPORTED_EXCHANGE_ALGORITHM)
    return backend.x25519_load_public_bytes(data)