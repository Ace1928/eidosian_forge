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
def test_fakenfs_rename_semantics(self):
    t = self.get_nfs_transport('.')
    self.build_tree(['from/', 'from/foo', 'to/', 'to/bar'], transport=t)
    self.assertRaises(errors.ResourceBusy, t.rename, 'from', 'to')