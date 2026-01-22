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
def verify_cert_signature(self) -> None:
    signature_key = self.signature_key()
    if isinstance(signature_key, ed25519.Ed25519PublicKey):
        signature_key.verify(bytes(self._signature), bytes(self._tbs_cert_body))
    elif isinstance(signature_key, ec.EllipticCurvePublicKey):
        r, data = _get_mpint(self._signature)
        s, data = _get_mpint(data)
        _check_empty(data)
        computed_sig = asym_utils.encode_dss_signature(r, s)
        hash_alg = _get_ec_hash_alg(signature_key.curve)
        signature_key.verify(computed_sig, bytes(self._tbs_cert_body), ec.ECDSA(hash_alg))
    else:
        assert isinstance(signature_key, rsa.RSAPublicKey)
        if self._inner_sig_type == _SSH_RSA:
            hash_alg = hashes.SHA1()
        elif self._inner_sig_type == _SSH_RSA_SHA256:
            hash_alg = hashes.SHA256()
        else:
            assert self._inner_sig_type == _SSH_RSA_SHA512
            hash_alg = hashes.SHA512()
        signature_key.verify(bytes(self._signature), bytes(self._tbs_cert_body), padding.PKCS1v15(), hash_alg)