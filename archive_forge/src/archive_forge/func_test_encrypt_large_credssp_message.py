import base64
import pytest
import struct
from winrm.encryption import Encryption
from winrm.exceptions import WinRMError
def test_encrypt_large_credssp_message():
    test_session = SessionTest()
    test_message = b'unencrypted message ' * 2048
    test_endpoint = 'http://testhost.com'
    message_chunks = [test_message[i:i + 16384] for i in range(0, len(test_message), 16384)]
    encryption = Encryption(test_session, 'credssp')
    actual = encryption.prepare_encrypted_request(test_session, test_endpoint, test_message)
    expected_encrypted_message1 = base64.b64encode(message_chunks[0])
    expected_encrypted_message2 = base64.b64encode(message_chunks[1])
    expected_encrypted_message3 = base64.b64encode(message_chunks[2])
    assert actual.headers == {'Content-Length': '55303', 'Content-Type': 'multipart/x-multi-encrypted;protocol="application/HTTP-CredSSP-session-encrypted";boundary="Encrypted Boundary"'}
    assert actual.body == b'--Encrypted Boundary\r\n\tContent-Type: application/HTTP-CredSSP-session-encrypted\r\n\tOriginalContent: type=application/soap+xml;charset=UTF-8;Length=16384\r\n--Encrypted Boundary\r\n\tContent-Type: application/octet-stream\r\n' + struct.pack('<i', 32) + expected_encrypted_message1 + b'--Encrypted Boundary\r\n\tContent-Type: application/HTTP-CredSSP-session-encrypted\r\n\tOriginalContent: type=application/soap+xml;charset=UTF-8;Length=16384\r\n--Encrypted Boundary\r\n\tContent-Type: application/octet-stream\r\n' + struct.pack('<i', 32) + expected_encrypted_message2 + b'--Encrypted Boundary\r\n\tContent-Type: application/HTTP-CredSSP-session-encrypted\r\n\tOriginalContent: type=application/soap+xml;charset=UTF-8;Length=8192\r\n--Encrypted Boundary\r\n\tContent-Type: application/octet-stream\r\n' + struct.pack('<i', 32) + expected_encrypted_message3 + b'--Encrypted Boundary--\r\n'