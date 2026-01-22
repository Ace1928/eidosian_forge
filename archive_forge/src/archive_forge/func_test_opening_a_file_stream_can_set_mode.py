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
def test_opening_a_file_stream_can_set_mode(self):
    t = self.get_transport()
    if t.is_readonly():
        self.assertRaises((TransportNotPossible, NotImplementedError), t.open_write_stream, 'foo')
        return
    if not t._can_roundtrip_unix_modebits():
        return

    def check_mode(name, mode, expected):
        handle = t.open_write_stream(name, mode=mode)
        handle.close()
        self.assertTransportMode(t, name, expected)
    check_mode('mode644', 420, 420)
    check_mode('mode666', 438, 438)
    check_mode('mode600', 384, 384)
    check_mode('nomode', None, 438 & ~osutils.get_umask())