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
def test_stop_gracefully_tells_handlers_to_stop(self):
    server, server_thread = self.make_server()
    client_sock = self.connect_to_server(server)
    self.say_hello(client_sock)
    server_handler, server_side_thread = server._active_connections[0]
    self.assertFalse(server_handler.finished)
    server._stop_gracefully()
    self.assertTrue(server_handler.finished)
    client_sock.close()
    self.connect_to_server_and_hangup(server)
    server_thread.join()