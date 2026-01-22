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
def test_socket_stream_with_two_requests(self):
    sample_request_bytes = b'command\n'
    server, client_sock = self.create_socket_context(None)
    first_protocol = SampleRequest(expected_bytes=sample_request_bytes)
    client_sock.sendall(sample_request_bytes * 2)
    server._serve_one_request(first_protocol)
    self.assertEqual(0, first_protocol.next_read_size())
    self.assertFalse(server.finished)
    second_protocol = SampleRequest(expected_bytes=sample_request_bytes)
    stream_still_open = server._serve_one_request(second_protocol)
    self.assertEqual(sample_request_bytes, second_protocol.accepted_bytes)
    self.assertFalse(server.finished)
    server._disconnect_client()
    self.assertEqual(b'', client_sock.recv(1))