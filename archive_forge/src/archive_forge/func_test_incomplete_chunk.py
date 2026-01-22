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
def test_incomplete_chunk(self):
    """When there are less bytes in the chunk than declared by the length,
        then we haven't finished reading yet.
        """
    decoder = protocol.ChunkedBodyDecoder()
    decoder.accept_bytes(b'chunked\n')
    chunk_length = b'8\n'
    three_bytes = b'123'
    decoder.accept_bytes(chunk_length + three_bytes)
    self.assertFalse(decoder.finished_reading)
    self.assertEqual(5 + 4, decoder.next_read_size(), "The next_read_size hint should be the number of missing bytes in this chunk plus 4 (the length of the end-of-body marker: 'END\\n')")
    self.assertEqual(None, decoder.read_next_chunk())