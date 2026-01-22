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
def test_relpath_at_root(self):
    t = self.get_transport()
    new_transport = t.clone('..')
    while new_transport.base != t.base:
        t = new_transport
        new_transport = t.clone('..')
    self.assertEqual('', t.relpath(t.base))
    self.assertEqual('foo/bar', t.relpath(t.base + 'foo/bar'))