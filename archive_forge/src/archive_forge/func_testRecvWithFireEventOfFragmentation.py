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
def testRecvWithFireEventOfFragmentation(self):
    sock = ws.WebSocket(fire_cont_frame=True)
    s = sock.sock = SockMock()
    s.add_packet(b'\x01\x8babcd#\x10\x06\x12\x08\x16\x1aD\x08\x11C')
    s.add_packet(b'\x00\x8babcd#\x10\x06\x12\x08\x16\x1aD\x08\x11C')
    s.add_packet(b'\x80\x8fabcd\x15\n\x06D\x12\r\x16\x08A\r\x05D\x16\x0b\x17')
    _, data = sock.recv_data()
    self.assertEqual(data, b'Brevity is ')
    _, data = sock.recv_data()
    self.assertEqual(data, b'Brevity is ')
    _, data = sock.recv_data()
    self.assertEqual(data, b'the soul of wit')
    s.add_packet(b'\x80\x8babcd#\x10\x06\x12\x08\x16\x1aD\x08\x11C')
    with self.assertRaises(ws.WebSocketException):
        sock.recv_data()
    with self.assertRaises(ws.WebSocketConnectionClosedException):
        sock.recv()