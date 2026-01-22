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
def test_trivial_response_decoding(self):
    """Smoke test for the simplest possible v3 response: empty headers,
        status byte, empty args, no body.
        """
    headers = b'\x00\x00\x00\x02de'
    response_status = b'oS'
    args = b's\x00\x00\x00\x02le'
    end = b'e'
    message_bytes = headers + response_status + args + end
    decoder, response_handler = self.make_logging_response_decoder()
    decoder.accept_bytes(message_bytes)
    self.assertEqual(0, decoder.next_read_size())
    self.assertEqual(b'', decoder.unused_data)
    self.assertEqual([('headers', {}), ('byte', b'S'), ('structure', ()), ('end',)], response_handler.event_log)