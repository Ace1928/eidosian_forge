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
def test_put_file_non_atomic(self):
    t = self.get_transport()
    if t.is_readonly():
        self.assertRaises(TransportNotPossible, t.put_file_non_atomic, 'a', BytesIO(b'some text for a\n'))
        return
    self.assertFalse(t.has('a'))
    t.put_file_non_atomic('a', BytesIO(b'some text for a\n'))
    self.assertTrue(t.has('a'))
    self.check_transport_contents(b'some text for a\n', t, 'a')
    t.put_file_non_atomic('a', BytesIO(b'new\ncontents for\na\n'))
    self.check_transport_contents(b'new\ncontents for\na\n', t, 'a')
    t.put_file_non_atomic('d', BytesIO(b'contents for\nd\n'))
    t.put_file_non_atomic('a', BytesIO(b''))
    self.check_transport_contents(b'contents for\nd\n', t, 'd')
    self.check_transport_contents(b'', t, 'a')
    self.assertRaises(NoSuchFile, t.put_file_non_atomic, 'no/such/path', BytesIO(b'contents\n'))
    self.assertRaises(NoSuchFile, t.put_file_non_atomic, 'dir/a', BytesIO(b'contents\n'))
    self.assertFalse(t.has('dir/a'))
    t.put_file_non_atomic('dir/a', BytesIO(b'contents for dir/a\n'), create_parent_dir=True)
    self.check_transport_contents(b'contents for dir/a\n', t, 'dir/a')
    self.assertRaises(NoSuchFile, t.put_file_non_atomic, 'not/there/a', BytesIO(b'contents\n'), create_parent_dir=True)