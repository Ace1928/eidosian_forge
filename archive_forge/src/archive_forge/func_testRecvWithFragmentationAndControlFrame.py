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
def testRecvWithFragmentationAndControlFrame(self):
    sock = ws.WebSocket()
    sock.set_mask_key(create_mask_key)
    s = sock.sock = SockMock()
    s.add_packet(b'\x01\x89abcd5\r\x0cD\x0c\x17\x00\x0cA')
    s.add_packet(b'\x89\x90abcd1\x0e\x06\x05\x12\x07C4.,$D\x15\n\n\x17')
    s.add_packet(b'\x80\x8fabcd\x0e\x04C\x05A\x05\x0c\x0b\x05B\x17\x0c\x08\x0c\x04')
    data = sock.recv()
    self.assertEqual(data, 'Too much of a good thing')
    with self.assertRaises(ws.WebSocketConnectionClosedException):
        sock.recv()
    self.assertEqual(s.sent[0], b'\x8a\x90abcd1\x0e\x06\x05\x12\x07C4.,$D\x15\n\n\x17')