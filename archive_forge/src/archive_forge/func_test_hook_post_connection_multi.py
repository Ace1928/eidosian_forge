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
def test_hook_post_connection_multi(self):
    """Fire post_connect hook once per unshared underlying connection"""
    log = []
    Transport.hooks.install_named_hook('post_connect', log.append, None)
    t1 = self.get_transport()
    t2 = t1.clone('.')
    t3 = self.get_transport()
    self.assertEqual([], log)
    t1.has('x')
    t2.has('x')
    t3.has('x')
    if isinstance(t1, RemoteTransport):
        self.assertEqual([t.get_smart_medium() for t in [t1, t3]], log)
    elif isinstance(t1, ConnectedTransport):
        self.assertEqual([t1, t3], log)
    else:
        self.assertEqual([], log)