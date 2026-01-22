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
def test_serve_conn_tracks_connections(self):
    server = _mod_server.SmartTCPServer(None, client_timeout=4.0)
    server_sock, client_sock = portable_socket_pair()
    server.serve_conn(server_sock, '-{}'.format(self.id()))
    self.assertEqual(1, len(server._active_connections))
    server._poll_active_connections()
    self.assertEqual(1, len(server._active_connections))
    client_sock.close()
    server._poll_active_connections(0.1)
    self.assertEqual(0, len(server._active_connections))