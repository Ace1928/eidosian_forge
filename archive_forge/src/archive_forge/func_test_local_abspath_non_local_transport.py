import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
def test_local_abspath_non_local_transport(self):
    t = memory.MemoryTransport()
    e = self.assertRaises(errors.NotLocalUrl, t.local_abspath, 't')
    self.assertEqual('memory:///t is not a local path.', str(e))