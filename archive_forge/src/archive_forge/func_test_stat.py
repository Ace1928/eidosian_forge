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
def test_stat(self):
    t = memory.MemoryTransport()
    t.put_bytes('foo', b'content')
    t.put_bytes('bar', b'phowar')
    self.assertEqual(7, t.stat('foo').st_size)
    self.assertEqual(6, t.stat('bar').st_size)