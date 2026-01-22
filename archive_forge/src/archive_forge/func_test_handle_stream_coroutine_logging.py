import socket
import subprocess
import sys
import textwrap
import unittest
from tornado import gen
from tornado.iostream import IOStream
from tornado.log import app_log
from tornado.tcpserver import TCPServer
from tornado.test.util import skipIfNonUnix
from tornado.testing import AsyncTestCase, ExpectLog, bind_unused_port, gen_test
from typing import Tuple
@gen_test
def test_handle_stream_coroutine_logging(self):

    class TestServer(TCPServer):

        @gen.coroutine
        def handle_stream(self, stream, address):
            yield stream.read_bytes(len(b'hello'))
            stream.close()
            1 / 0
    server = client = None
    try:
        sock, port = bind_unused_port()
        server = TestServer()
        server.add_socket(sock)
        client = IOStream(socket.socket())
        with ExpectLog(app_log, 'Exception in callback'):
            yield client.connect(('localhost', port))
            yield client.write(b'hello')
            yield client.read_until_close()
            yield gen.moment
    finally:
        if server is not None:
            server.stop()
        if client is not None:
            client.close()