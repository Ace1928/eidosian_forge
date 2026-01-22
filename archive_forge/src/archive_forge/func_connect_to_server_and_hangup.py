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
def connect_to_server_and_hangup(self, server):
    """Connect to the server, and then hang up.
        That way it doesn't sit waiting for 'accept()' to timeout.
        """
    if server._stopped.is_set():
        return
    try:
        client_sock = self.connect_to_server(server)
        client_sock.close()
    except OSError as e:
        pass