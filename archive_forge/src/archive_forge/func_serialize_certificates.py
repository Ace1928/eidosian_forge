from __future__ import annotations
import email.base64mime
import email.generator
import email.message
import email.policy
import io
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import pkcs7 as rust_pkcs7
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.utils import _check_byteslike
def serialize_certificates(certs: typing.List[x509.Certificate], encoding: serialization.Encoding) -> bytes:
    return rust_pkcs7.serialize_certificates(certs, encoding)