import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_server_fails_while_serving_or_stopping(self):

    class CantConnect(Exception):
        pass

    class FailingConnectionHandler(TCPConnectionHandler):

        def handle(self):
            raise CantConnect()
    server = self.get_server(connection_handler_class=FailingConnectionHandler)
    client = self.get_client()
    client.connect((server.host, server.port))
    client.write(b'ping\n')
    try:
        self.assertEqual(b'', client.read())
    except OSError as e:
        WSAECONNRESET = 10054
        if e.errno in (WSAECONNRESET,):
            pass
    self.assertRaises(CantConnect, server.stop_server)