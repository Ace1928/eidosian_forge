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
def test_append_file(self):
    t = self.get_transport()
    if t.is_readonly():
        self.assertRaises(TransportNotPossible, t.append_file, 'a', 'add\nsome\nmore\ncontents\n')
        return
    t.put_bytes('a', b'diff\ncontents for\na\n')
    t.put_bytes('b', b'contents\nfor b\n')
    self.assertEqual(20, t.append_file('a', BytesIO(b'add\nsome\nmore\ncontents\n')))
    self.check_transport_contents(b'diff\ncontents for\na\nadd\nsome\nmore\ncontents\n', t, 'a')
    self.assertRaises(NoSuchFile, t.append_file, 'missing/path', BytesIO(b'content'))
    self.assertEqual(0, t.append_file('c', BytesIO(b'some text\nfor a missing file\n')))
    self.check_transport_contents(b'some text\nfor a missing file\n', t, 'c')