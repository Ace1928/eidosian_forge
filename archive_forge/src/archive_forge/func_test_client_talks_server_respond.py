import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_client_talks_server_respond(self):
    server = self.get_server()
    client = self.get_client()
    client.connect((server.host, server.port))
    self.assertIs(None, client.write(b'ping\n'))
    resp = client.read()
    self.assertClientAddr(client, server, 0)
    self.assertEqual(b'pong\n', resp)