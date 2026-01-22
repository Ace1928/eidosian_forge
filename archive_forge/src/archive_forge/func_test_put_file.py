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
def test_put_file(self):
    t = self.get_transport()
    if t.is_readonly():
        self.assertRaises(TransportNotPossible, t.put_file, 'a', BytesIO(b'some text for a\n'))
        return
    result = t.put_file('a', BytesIO(b'some text for a\n'))
    self.assertEqual(16, result)
    self.assertTrue(t.has('a'))
    self.check_transport_contents(b'some text for a\n', t, 'a')
    result = t.put_file('a', BytesIO(b'new\ncontents for\na\n'))
    self.assertEqual(19, result)
    self.check_transport_contents(b'new\ncontents for\na\n', t, 'a')
    self.assertRaises(NoSuchFile, t.put_file, 'path/doesnt/exist/c', BytesIO(b'contents'))