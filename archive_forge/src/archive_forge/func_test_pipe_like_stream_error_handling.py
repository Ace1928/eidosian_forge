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
def test_pipe_like_stream_error_handling(self):
    from io import BytesIO
    to_server = BytesIO(b'')
    from_server = BytesIO()
    self.closed = False

    def close():
        self.closed = True
    from_server.close = close
    server = self.create_pipe_medium(to_server, from_server, None)
    fake_protocol = ErrorRaisingProtocol(Exception('boom'))
    server._serve_one_request(fake_protocol)
    self.assertEqual(b'', from_server.getvalue())
    self.assertTrue(self.closed)
    self.assertTrue(server.finished)