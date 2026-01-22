import os
import os.path
import socket
import unittest
from base64 import decodebytes as base64decode
import websocket as ws
from websocket._handshake import _create_sec_websocket_key
from websocket._handshake import _validate as _validate_header
from websocket._http import read_headers
from websocket._utils import validate_utf8
@unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
def test_http_SSL(self):
    websock1 = ws.WebSocket(sslopt={'cert_chain': ssl.get_default_verify_paths().capath}, enable_multithread=False)
    self.assertRaises(ValueError, websock1.connect, 'wss://api.bitfinex.com/ws/2')
    websock2 = ws.WebSocket(sslopt={'certfile': 'myNonexistentCertFile'})
    self.assertRaises(FileNotFoundError, websock2.connect, 'wss://api.bitfinex.com/ws/2')