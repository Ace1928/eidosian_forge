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
def test_delete_tree(self):
    t = self.get_transport()
    if t.is_readonly():
        self.assertRaises(TransportNotPossible, t.delete_tree, 'missing')
        return
    t.mkdir('adir')
    try:
        t.delete_tree('adir')
    except TransportNotPossible:
        return
    self.assertRaises(NoSuchFile, t.stat, 'adir')
    self.build_tree(['adir/', 'adir/file', 'adir/subdir/', 'adir/subdir/file', 'adir/subdir2/', 'adir/subdir2/file'], transport=t)
    t.delete_tree('adir')
    self.assertRaises(NoSuchFile, t.stat, 'adir')