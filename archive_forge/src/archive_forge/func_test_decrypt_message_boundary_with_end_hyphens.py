import base64
import pytest
import struct
from winrm.encryption import Encryption
from winrm.exceptions import WinRMError
def test_decrypt_message_boundary_with_end_hyphens():
    test_session = SessionTest()
    test_encrypted_message = b'dW5lbmNyeXB0ZWQgbWVzc2FnZQ=='
    test_signature = b'1234'
    test_signature_length = struct.pack('<i', len(test_signature))
    test_message = b'--Encrypted Boundary\r\n\tContent-Type: application/HTTP-SPNEGO-session-encrypted\r\n\tOriginalContent: type=application/soap+xml;charset=UTF-8;Length=19\r\n--Encrypted Boundary\r\n\tContent-Type: application/octet-stream\r\n' + test_signature_length + test_signature + test_encrypted_message + b'--Encrypted Boundary--\r\n'
    test_response = ResponseTest('protocol="application/HTTP-SPNEGO-session-encrypted"', test_message)
    encryption = Encryption(test_session, 'ntlm')
    actual = encryption.parse_encrypted_response(test_response)
    assert actual == b'unencrypted message'