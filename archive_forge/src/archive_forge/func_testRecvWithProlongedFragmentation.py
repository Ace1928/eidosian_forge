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
def testRecvWithProlongedFragmentation(self):
    sock = ws.WebSocket()
    s = sock.sock = SockMock()
    s.add_packet(b'\x01\x9babcd.\x0c\x00\x01A\x0f\x0c\x16\x04B\x16\n\x15\rC\x10\t\x07C\x06\x13\x07\x02\x07\tNC')
    s.add_packet(b'\x00\x8eabcd\x05\x07\x02\x16A\x04\x11\r\x04\x0c\x07\x17MB')
    s.add_packet(b'\x80\x89abcd\x0e\x0c\x00\x01A\x0f\x0c\x16\x04')
    data = sock.recv()
    self.assertEqual(data, 'Once more unto the breach, dear friends, once more')
    with self.assertRaises(ws.WebSocketConnectionClosedException):
        sock.recv()