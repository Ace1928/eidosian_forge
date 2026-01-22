import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_sets_client_timeout(self):
    server = test_server.TestingSmartServer(('localhost', 0), None, None, root_client_path='/no-such-client/path')
    self.assertEqual(test_server._DEFAULT_TESTING_CLIENT_TIMEOUT, server._client_timeout)
    sock = socket.socket()
    h = server._make_handler(sock)
    self.assertEqual(test_server._DEFAULT_TESTING_CLIENT_TIMEOUT, h._client_timeout)