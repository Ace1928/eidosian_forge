import base64
import pytest
import struct
from winrm.encryption import Encryption
from winrm.exceptions import WinRMError
def test_get_credssp_trailer_length_gcm():
    test_session = SessionTest()
    encryption = Encryption(test_session, 'credssp')
    expected = 16
    actual = encryption._get_credssp_trailer_length(30, 'ECDHE-RSA-AES128-GCM-SHA256')
    assert actual == expected