import os
import stat
import sys
from io import BytesIO
from .. import errors, osutils, pyutils, tests
from .. import transport as _mod_transport
from .. import urlutils
from ..errors import ConnectionError, PathError, TransportNotPossible
from ..osutils import getcwd
from ..transport import (ConnectedTransport, FileExists, NoSuchFile, Transport,
from ..transport.memory import MemoryTransport
from ..transport.remote import RemoteTransport
from . import TestNotApplicable, TestSkipped, multiply_tests, test_server
from .test_transport import TestTransportImplementation
def test_opening_a_file_stream_creates_file(self):
    t = self.get_transport()
    if t.is_readonly():
        return
    handle = t.open_write_stream('foo')
    try:
        self.assertEqual(b'', t.get_bytes('foo'))
    finally:
        handle.close()