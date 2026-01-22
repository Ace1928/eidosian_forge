import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
def test_key(self):
    KEY = b'-----BEGIN PRIVATE KEY-----\n        MIIBCgKCAQEA4ej0p7bQ7L/r4rVGUz9RN4VQWoej1Bg1mYWIDYslvKrk1gpj7wZg\n        /fy3ZpsL7WqgsZS7Q+0VRK8gKfqkxg5OYQIDAQAB\n        -----END PRIVATE KEY-----'
    RSA_KEY = b'-----BEGIN RSA PRIVATE KEY-----\n        MIIBCgKCAQEA4ej0p7bQ7L/r4rVGUz9RN4VQWoej1Bg1mYWIDYslvKrk1gpj7wZg\n        /fy3ZpsL7WqgsZS7Q+0VRK8gKfqkxg5OYQIDAQAB\n        -----END RSA PRIVATE KEY-----'
    EC_KEY = b'-----BEGIN EC PRIVATE KEY-----\n        MIIBCgKCAQEA4ej0p7bQ7L/r4rVGUz9RN4VQWoej1Bg1mYWIDYslvKrk1gpj7wZg\n        /fy3ZpsL7WqgsZS7Q+0VRK8gKfqkxg5OYQIDAQAB\n        -----END EC PRIVATE KEY-----'
    check_cert_and_key(pytest.public_cert_bytes + KEY, pytest.public_cert_bytes, KEY)
    check_cert_and_key(pytest.public_cert_bytes + RSA_KEY, pytest.public_cert_bytes, RSA_KEY)
    check_cert_and_key(pytest.public_cert_bytes + EC_KEY, pytest.public_cert_bytes, EC_KEY)