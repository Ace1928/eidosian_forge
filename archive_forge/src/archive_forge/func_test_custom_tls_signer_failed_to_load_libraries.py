import base64
import ctypes
import os
import mock
import pytest  # type: ignore
from requests.packages.urllib3.util.ssl_ import create_urllib3_context  # type: ignore
import urllib3.contrib.pyopenssl  # type: ignore
from google.auth import exceptions
from google.auth.transport import _custom_tls_signer
def test_custom_tls_signer_failed_to_load_libraries():
    with pytest.raises(exceptions.MutualTLSChannelError) as excinfo:
        signer_object = _custom_tls_signer.CustomTlsSigner(INVALID_ENTERPRISE_CERT_FILE)
        signer_object.load_libraries()
    assert excinfo.match('enterprise cert file is invalid')