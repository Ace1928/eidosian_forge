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
def test_get_directory_read_gives_ReadError(self):
    """consistent errors for read() on a file returned by get()."""
    t = self.get_transport()
    if t.is_readonly():
        self.build_tree(['a directory/'])
    else:
        t.mkdir('a%20directory')
    try:
        a_file = t.get('a%20directory')
    except (errors.PathError, errors.RedirectRequested):
        return
    try:
        a_file.read()
    except errors.ReadError:
        pass