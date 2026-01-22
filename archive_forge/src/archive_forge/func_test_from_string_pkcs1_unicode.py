import base64
import json
import os
from cryptography.hazmat.primitives.asymmetric import ec
import pytest  # type: ignore
from google.auth import _helpers
from google.auth.crypt import base
from google.auth.crypt import es256
def test_from_string_pkcs1_unicode(self):
    key_bytes = _helpers.from_bytes(PKCS1_KEY_BYTES)
    signer = es256.ES256Signer.from_string(key_bytes)
    assert isinstance(signer, es256.ES256Signer)
    assert isinstance(signer._key, ec.EllipticCurvePrivateKey)