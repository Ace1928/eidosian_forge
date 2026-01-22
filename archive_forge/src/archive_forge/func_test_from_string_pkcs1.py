import base64
import json
import os
from cryptography.hazmat.primitives.asymmetric import ec
import pytest  # type: ignore
from google.auth import _helpers
from google.auth.crypt import base
from google.auth.crypt import es256
def test_from_string_pkcs1(self):
    signer = es256.ES256Signer.from_string(PKCS1_KEY_BYTES)
    assert isinstance(signer, es256.ES256Signer)
    assert isinstance(signer._key, ec.EllipticCurvePrivateKey)