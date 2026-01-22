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
def test_append_bytes_mode(self):
    t = self.get_transport()
    if t.is_readonly():
        self.assertRaises(TransportNotPossible, t.append_bytes, 'f', b'f', mode=None)
        return
    t.append_bytes('f', b'f', mode=None)