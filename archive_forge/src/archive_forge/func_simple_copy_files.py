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
def simple_copy_files(transport_from, transport_to):
    files = ['a', 'b', 'c', 'd']
    self.build_tree(files, transport=transport_from)
    self.assertEqual(4, transport_from.copy_to(files, transport_to))
    for f in files:
        self.check_transport_contents(transport_to.get_bytes(f), transport_from, f)