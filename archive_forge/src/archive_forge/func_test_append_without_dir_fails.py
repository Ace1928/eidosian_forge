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
def test_append_without_dir_fails(self):
    t = memory.MemoryTransport()
    self.assertRaises(NoSuchFile, t.append_bytes, 'dir/path', b'content')