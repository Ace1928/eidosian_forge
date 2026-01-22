import os
import os.path
import socket
import ssl
import unittest
import websocket
import websocket as ws
from websocket._http import (
@unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
@unittest.skipUnless(TEST_WITH_PROXY, 'This test requires a HTTP proxy to be running on port 8899')
@unittest.skipUnless(TEST_WITH_LOCAL_SERVER, 'Tests using local websocket server are disabled')
def testProxyConnect(self):
    ws = websocket.WebSocket()
    ws.connect(f'ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}', http_proxy_host='127.0.0.1', http_proxy_port='8899', proxy_type='http')
    ws.send('Hello, Server')
    server_response = ws.recv()
    self.assertEqual(server_response, 'Hello, Server')
    self.assertEqual(_get_addrinfo_list('api.bitfinex.com', 443, True, proxy_info(http_proxy_host='127.0.0.1', http_proxy_port='8899', proxy_type='http')), (socket.getaddrinfo('127.0.0.1', 8899, 0, socket.SOCK_STREAM, socket.SOL_TCP), True, None))
    self.assertEqual(connect('wss://api.bitfinex.com/ws/2', OptsList(), proxy_info(http_proxy_host='127.0.0.1', http_proxy_port=8899, proxy_type='http'), None)[1], ('api.bitfinex.com', 443, '/ws/2'))