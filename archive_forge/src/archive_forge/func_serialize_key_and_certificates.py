from __future__ import annotations
import typing
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives._serialization import PBES as PBES
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes
def serialize_key_and_certificates(name: typing.Optional[bytes], key: typing.Optional[PKCS12PrivateKeyTypes], cert: typing.Optional[x509.Certificate], cas: typing.Optional[typing.Iterable[_PKCS12CATypes]], encryption_algorithm: serialization.KeySerializationEncryption) -> bytes:
    if key is not None and (not isinstance(key, (rsa.RSAPrivateKey, dsa.DSAPrivateKey, ec.EllipticCurvePrivateKey, ed25519.Ed25519PrivateKey, ed448.Ed448PrivateKey))):
        raise TypeError('Key must be RSA, DSA, EllipticCurve, ED25519, or ED448 private key, or None.')
    if cert is not None and (not isinstance(cert, x509.Certificate)):
        raise TypeError('cert must be a certificate or None')
    if cas is not None:
        cas = list(cas)
        if not all((isinstance(val, (x509.Certificate, PKCS12Certificate)) for val in cas)):
            raise TypeError('all values in cas must be certificates')
    if not isinstance(encryption_algorithm, serialization.KeySerializationEncryption):
        raise TypeError('Key encryption algorithm must be a KeySerializationEncryption instance')
    if key is None and cert is None and (not cas):
        raise ValueError('You must supply at least one of key, cert, or cas')
    from cryptography.hazmat.backends.openssl.backend import backend
    return backend.serialize_key_and_certificates_to_pkcs12(name, key, cert, cas, encryption_algorithm)