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
def test_clone_to_root(self):
    orig_transport = self.get_transport()
    root_transport = orig_transport
    new_transport = root_transport.clone('..')
    self.assertTrue(len(new_transport.base) < len(root_transport.base) or new_transport.base == root_transport.base)
    while new_transport.base != root_transport.base:
        root_transport = new_transport
        new_transport = root_transport.clone('..')
        self.assertTrue(len(new_transport.base) < len(root_transport.base) or new_transport.base == root_transport.base)
    self.assertEqual(root_transport.base, orig_transport.clone('/').base)
    self.assertEqual(orig_transport.abspath('/'), root_transport.base)
    self.assertEqual(root_transport.base[-1], '/')