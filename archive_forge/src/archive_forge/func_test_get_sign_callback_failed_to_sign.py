import base64
import ctypes
import os
import mock
import pytest  # type: ignore
from requests.packages.urllib3.util.ssl_ import create_urllib3_context  # type: ignore
import urllib3.contrib.pyopenssl  # type: ignore
from google.auth import exceptions
from google.auth.transport import _custom_tls_signer
def test_get_sign_callback_failed_to_sign():
    mock_sig_len = 0
    mock_signer_lib = mock.MagicMock()
    mock_signer_lib.SignForPython.return_value = mock_sig_len
    sign_callback = _custom_tls_signer.get_sign_callback(mock_signer_lib, FAKE_ENTERPRISE_CERT_FILE_PATH)
    to_be_signed = ctypes.POINTER(ctypes.c_ubyte)()
    to_be_signed_len = 4
    returned_sig_array = ctypes.c_ubyte()
    mock_sig_array = ctypes.byref(returned_sig_array)
    returned_sign_len = ctypes.c_ulong()
    mock_sig_len_array = ctypes.byref(returned_sign_len)
    sign_callback(mock_sig_array, mock_sig_len_array, to_be_signed, to_be_signed_len)
    assert not sign_callback(mock_sig_array, mock_sig_len_array, to_be_signed, to_be_signed_len)