import doctest
import errno
import os
import socket
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional, Type
from testtools.matchers import DocTestMatches
import breezy
from ... import controldir, debug, errors, osutils, tests
from ... import transport as _mod_transport
from ... import urlutils
from ...tests import features, test_server
from ...transport import local, memory, remote, ssh
from ...transport.http import urllib
from .. import bzrdir
from ..remote import UnknownErrorFromSmartServer
from ..smart import client, medium, message, protocol
from ..smart import request as _mod_request
from ..smart import server as _mod_server
from ..smart import vfs
from . import test_smart
def test_graceful_shutdown_waits_for_clients_to_stop(self):
    server, server_thread = self.make_server()
    server.backing_transport.put_bytes('bigfile', b'a' * 1024 * 1024)
    client_sock = self.connect_to_server(server)
    self.say_hello(client_sock)
    _, server_side_thread = server._active_connections[0]
    client_medium = medium.SmartClientAlreadyConnectedSocketMedium('base', client_sock)
    client_client = client._SmartClient(client_medium)
    resp, response_handler = client_client.call_expecting_body(b'get', b'bigfile')
    self.assertEqual((b'ok',), resp)
    server._stop_gracefully()
    self.connect_to_server_and_hangup(server)
    server._stopped.wait()
    self.assertRaises(socket.error, self.connect_to_server, server)
    response_handler.read_body_bytes()
    client_sock.close()
    server_side_thread.join()
    server_thread.join()
    self.assertTrue(server._fully_stopped.is_set())
    log = self.get_log()
    self.assertThat(log, DocTestMatches("    INFO  Requested to stop gracefully\n... Stopping SmartServerSocketStreamMedium(client=('127.0.0.1', ...\n", flags=doctest.ELLIPSIS | doctest.REPORT_UDIFF))