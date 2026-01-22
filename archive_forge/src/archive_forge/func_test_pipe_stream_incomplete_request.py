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
def test_pipe_stream_incomplete_request(self):
    """The medium should still construct the right protocol version even if
        the initial read only reads part of the request.

        Specifically, it should correctly read the protocol version line even
        if the partial read doesn't end in a newline.  An older, naive
        implementation of _get_line in the server used to have a bug in that
        case.
        """
    incomplete_request_bytes = protocol.REQUEST_VERSION_TWO + b'hel'
    rest_of_request_bytes = b'lo\n'
    expected_response = protocol.RESPONSE_VERSION_TWO + b'success\nok\x012\n'
    to_server, to_server_w = os.pipe()
    from_server_r, from_server = os.pipe()
    to_server = os.fdopen(to_server, 'rb', 0)
    to_server_w = os.fdopen(to_server_w, 'wb', 0)
    from_server_r = os.fdopen(from_server_r, 'rb', 0)
    from_server = os.fdopen(from_server, 'wb', 0)
    server = self.create_pipe_medium(to_server, from_server, None)
    to_server_w.write(incomplete_request_bytes)
    server_protocol = server._build_protocol()
    to_server_w.write(rest_of_request_bytes)
    server._serve_one_request(server_protocol)
    to_server_w.close()
    from_server.close()
    self.assertEqual(expected_response, from_server_r.read(), "Not a version 2 response to 'hello' request.")
    self.assertEqual(b'', from_server_r.read(1))
    from_server_r.close()
    to_server.close()