import json
import os
from cryptography.hazmat.primitives.asymmetric import rsa
import pytest  # type: ignore
from google.auth import _helpers
from google.auth.crypt import _cryptography_rsa
from google.auth.crypt import base
def test_from_string_pkcs8_unicode(self):
    key_bytes = _helpers.from_bytes(PKCS8_KEY_BYTES)
    signer = _cryptography_rsa.RSASigner.from_string(key_bytes)
    assert isinstance(signer, _cryptography_rsa.RSASigner)
    assert isinstance(signer._key, rsa.RSAPrivateKey)