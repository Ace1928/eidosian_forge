from __future__ import annotations
import binascii
import enum
import os
import re
import typing
import warnings
from base64 import encodebytes as _base64_encode
from dataclasses import dataclass
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.serialization import (
def load_ssh_public_key(data: bytes, backend: typing.Any=None) -> SSHPublicKeyTypes:
    cert_or_key = _load_ssh_public_identity(data, _legacy_dsa_allowed=True)
    public_key: SSHPublicKeyTypes
    if isinstance(cert_or_key, SSHCertificate):
        public_key = cert_or_key.public_key()
    else:
        public_key = cert_or_key
    if isinstance(public_key, dsa.DSAPublicKey):
        warnings.warn('SSH DSA keys are deprecated and will be removed in a future release.', utils.DeprecatedIn40, stacklevel=2)
    return public_key