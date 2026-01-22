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
def make_server(self):
    """Create a SmartTCPServer that we can exercise.

        Note: we don't use SmartTCPServer_for_testing because the testing
        version overrides lots of functionality like 'serve', and we want to
        test the raw service.

        This will start the server in another thread, and wait for it to
        indicate it has finished starting up.

        :return: (server, server_thread)
        """
    t = _mod_transport.get_transport_from_url('memory:///')
    server = _mod_server.SmartTCPServer(t, client_timeout=4.0)
    server._ACCEPT_TIMEOUT = 0.1
    server.start_server('127.0.0.1', 0)
    server_thread = threading.Thread(target=server.serve, args=(self.id(),))
    server_thread.start()
    self.addCleanup(server._stop_gracefully)
    server._started.wait()
    return (server, server_thread)