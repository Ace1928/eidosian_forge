import os
from google.auth import crypt
def test_verify_signature_failure():
    to_sign = b'foo'
    signer = crypt.RSASigner.from_string(PRIVATE_KEY_BYTES)
    signature = signer.sign(to_sign)
    assert not crypt.verify_signature(to_sign, signature, OTHER_CERT_BYTES)