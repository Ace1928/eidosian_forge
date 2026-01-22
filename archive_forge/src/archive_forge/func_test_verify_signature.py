import os
from google.auth import crypt
def test_verify_signature():
    to_sign = b'foo'
    signer = crypt.RSASigner.from_string(PRIVATE_KEY_BYTES)
    signature = signer.sign(to_sign)
    assert crypt.verify_signature(to_sign, signature, PUBLIC_CERT_BYTES)
    assert crypt.verify_signature(to_sign, signature, [OTHER_CERT_BYTES, PUBLIC_CERT_BYTES])