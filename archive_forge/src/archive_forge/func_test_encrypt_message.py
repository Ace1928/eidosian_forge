import base64
import pytest
import struct
from winrm.encryption import Encryption
from winrm.exceptions import WinRMError
def test_encrypt_message():
    test_session = SessionTest()
    test_message = b'unencrypted message'
    test_endpoint = b'endpoint'
    encryption = Encryption(test_session, 'ntlm')
    actual = encryption.prepare_encrypted_request(test_session, test_endpoint, test_message)
    expected_encrypted_message = b'dW5lbmNyeXB0ZWQgbWVzc2FnZQ=='
    expected_signature = b'1234'
    signature_length = struct.pack('<i', len(expected_signature))
    assert actual.headers == {'Content-Length': '272', 'Content-Type': 'multipart/encrypted;protocol="application/HTTP-SPNEGO-session-encrypted";boundary="Encrypted Boundary"'}
    assert actual.body == b'--Encrypted Boundary\r\n\tContent-Type: application/HTTP-SPNEGO-session-encrypted\r\n\tOriginalContent: type=application/soap+xml;charset=UTF-8;Length=19\r\n--Encrypted Boundary\r\n\tContent-Type: application/octet-stream\r\n' + signature_length + expected_signature + expected_encrypted_message + b'--Encrypted Boundary--\r\n'