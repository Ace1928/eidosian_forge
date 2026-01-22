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
def test_put_and_get(self):
    t = memory.MemoryTransport()
    t.put_file('path', BytesIO(b'content'))
    self.assertEqual(t.get('path').read(), b'content')
    t.put_bytes('path', b'content')
    self.assertEqual(t.get('path').read(), b'content')