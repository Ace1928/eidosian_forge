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
def test_server_closes_listening_sock_on_shutdown_after_request(self):
    """The server should close its listening socket when it's stopped."""
    self.start_server()
    server_url = self.server.get_url()
    self.transport.has('.')
    self.stop_server()
    t = remote.RemoteTCPTransport(server_url)
    self.assertRaises(errors.ConnectionError, t.has, '.')