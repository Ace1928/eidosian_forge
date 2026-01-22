import errno
import socket
import socketserver
import threading
from breezy import osutils, tests
from breezy.tests import test_server
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_server_crash_while_responding(self):
    caught = threading.Event()
    caught.clear()
    self.connection_thread = None

    class FailToRespond(Exception):
        pass

    class FailingDuringResponseHandler(TCPConnectionHandler):

        def handle_connection(request):
            request.readline()
            self.connection_thread = threading.currentThread()
            self.connection_thread.set_sync_event(caught)
            raise FailToRespond()
    server = self.get_server(connection_handler_class=FailingDuringResponseHandler)
    client = self.get_client()
    client.connect((server.host, server.port))
    client.write(b'ping\n')
    caught.wait()
    self.assertEqual(b'', client.read())
    self.assertRaises(FailToRespond, self.connection_thread.pending_exception)