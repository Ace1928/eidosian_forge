import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
def check_cert_and_key(content, expected_cert, expected_key):
    success = True
    cert_match = re.findall(_mtls_helper._CERT_REGEX, content)
    success = success and len(cert_match) == 1 and (cert_match[0] == expected_cert)
    key_match = re.findall(_mtls_helper._KEY_REGEX, content)
    success = success and len(key_match) == 1 and (key_match[0] == expected_key)
    return success