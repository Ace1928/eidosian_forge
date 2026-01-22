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
def test_pipe_wait_for_bytes_with_timeout_no_data(self):
    r_server, w_client = os.pipe()
    with os.fdopen(r_server, 'rb') as rf_server:
        server = self.create_pipe_medium(rf_server, None, None)
        if sys.platform == 'win32':
            server._wait_for_bytes_with_timeout(0.01)
        else:
            self.assertRaises(errors.ConnectionTimeout, server._wait_for_bytes_with_timeout, 0.01)
        os.close(w_client)
        data = server.read_bytes(5)
        self.assertEqual(b'', data)