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
def test_get_unknown_file(self):
    t = self.get_transport()
    files = ['a', 'b']
    contents = [b'contents of a\n', b'contents of b\n']
    self.build_tree(files, transport=t, line_endings='binary')
    self.assertRaises(NoSuchFile, t.get, 'c')

    def iterate_and_close(func, *args):
        for f in func(*args):
            content = f.read()
            f.close()