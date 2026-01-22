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
def test_reuse_connection_for_various_paths(self):
    t = self.get_transport()
    if not isinstance(t, ConnectedTransport):
        raise TestSkipped('not a connected transport')
    t.has('surely_not')
    self.assertIsNot(None, t._get_connection())
    subdir = t._reuse_for(t.base + 'whatever/but/deep/down/the/path')
    self.assertIsNot(t, subdir)
    self.assertIs(t._get_connection(), subdir._get_connection())
    home = subdir._reuse_for(t.base + 'home')
    self.assertIs(t._get_connection(), home._get_connection())
    self.assertIs(subdir._get_connection(), home._get_connection())