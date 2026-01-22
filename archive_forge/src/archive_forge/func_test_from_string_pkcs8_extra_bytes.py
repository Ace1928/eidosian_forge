import json
import os
import mock
from pyasn1_modules import pem  # type: ignore
import pytest  # type: ignore
import rsa  # type: ignore
import six
from google.auth import _helpers
from google.auth.crypt import _python_rsa
from google.auth.crypt import base
def test_from_string_pkcs8_extra_bytes(self):
    key_bytes = PKCS8_KEY_BYTES
    _, pem_bytes = pem.readPemBlocksFromFile(six.StringIO(_helpers.from_bytes(key_bytes)), _python_rsa._PKCS8_MARKER)
    key_info, remaining = (None, 'extra')
    decode_patch = mock.patch('pyasn1.codec.der.decoder.decode', return_value=(key_info, remaining), autospec=True)
    with decode_patch as decode:
        with pytest.raises(ValueError):
            _python_rsa.RSASigner.from_string(key_bytes)
        decode.assert_called_once_with(pem_bytes, asn1Spec=_python_rsa._PKCS8_SPEC)