import base64
import ctypes
import os
import mock
import pytest  # type: ignore
from requests.packages.urllib3.util.ssl_ import create_urllib3_context  # type: ignore
import urllib3.contrib.pyopenssl  # type: ignore
from google.auth import exceptions
from google.auth.transport import _custom_tls_signer
def test_custom_tls_signer_fail_to_offload():
    offload_lib = mock.MagicMock()
    signer_lib = mock.MagicMock()
    with mock.patch('google.auth.transport._custom_tls_signer.load_signer_lib') as load_signer_lib:
        with mock.patch('google.auth.transport._custom_tls_signer.load_offload_lib') as load_offload_lib:
            load_offload_lib.return_value = offload_lib
            load_signer_lib.return_value = signer_lib
            signer_object = _custom_tls_signer.CustomTlsSigner(ENTERPRISE_CERT_FILE)
            signer_object.load_libraries()
    offload_lib.ConfigureSslContext.return_value = 0
    with pytest.raises(exceptions.MutualTLSChannelError) as excinfo:
        with mock.patch('google.auth.transport._custom_tls_signer.get_cert') as get_cert:
            with mock.patch('google.auth.transport._custom_tls_signer.get_sign_callback'):
                get_cert.return_value = b'mock_cert'
                signer_object.set_up_custom_key()
                signer_object.attach_to_ssl_context(create_urllib3_context())
    assert excinfo.match('failed to configure SSL context')