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
def test_non_conventional_request(self):
    """ConventionalRequestHandler (the default message handler on the
        server side) will reject an unconventional message, but still consume
        all the bytes of that message and signal when it has done so.

        This is what allows a server to continue to accept requests after the
        client sends a completely unrecognised request.
        """
    invalid_request = protocol.MESSAGE_VERSION_THREE + b'\x00\x00\x00\x02de' + b'oX' + b'oX' + b'e'
    to_server = BytesIO(invalid_request)
    from_server = BytesIO()
    transport = memory.MemoryTransport('memory:///')
    server = medium.SmartServerPipeStreamMedium(to_server, from_server, transport, timeout=4.0)
    proto = server._build_protocol()
    message_handler = proto.message_handler
    server._serve_one_request(proto)
    self.assertEqual(b'', to_server.read())
    self.assertEqual(b'', proto.unused_data)
    self.assertEqual(0, proto.next_read_size())