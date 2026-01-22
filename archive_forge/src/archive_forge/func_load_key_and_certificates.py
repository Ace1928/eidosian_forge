from __future__ import annotations
import typing
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives._serialization import PBES as PBES
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes
def load_key_and_certificates(data: bytes, password: typing.Optional[bytes], backend: typing.Any=None) -> typing.Tuple[typing.Optional[PrivateKeyTypes], typing.Optional[x509.Certificate], typing.List[x509.Certificate]]:
    from cryptography.hazmat.backends.openssl.backend import backend as ossl
    return ossl.load_key_and_certificates_from_pkcs12(data, password)