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
def test_clone_preserve_info(self):
    t1 = self.get_transport()
    if not isinstance(t1, ConnectedTransport):
        raise TestSkipped('not a connected transport')
    t2 = t1.clone('subdir')
    self.assertEqual(t1._parsed_url.scheme, t2._parsed_url.scheme)
    self.assertEqual(t1._parsed_url.user, t2._parsed_url.user)
    self.assertEqual(t1._parsed_url.password, t2._parsed_url.password)
    self.assertEqual(t1._parsed_url.host, t2._parsed_url.host)
    self.assertEqual(t1._parsed_url.port, t2._parsed_url.port)