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
def test_win32_abspath(self):
    if sys.platform != 'win32':
        raise TestSkipped('Testing drive letters in abspath implemented only for win32')
    transport = _mod_transport.get_transport_from_url('file:///')
    self.assertEqual(transport.abspath('/'), 'file:///')
    transport = _mod_transport.get_transport_from_url('file:///C:/')
    self.assertEqual(transport.abspath('/'), 'file:///C:/')