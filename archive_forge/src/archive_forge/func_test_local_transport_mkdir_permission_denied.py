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
def test_local_transport_mkdir_permission_denied(self):
    here = osutils.abspath('.')
    t = transport.get_transport(here)

    def fake_chmod(path, mode):
        e = OSError('permission denied')
        e.errno = errno.EPERM
        raise e
    self.overrideAttr(os, 'chmod', fake_chmod)
    t.mkdir('test')
    t.mkdir('test2', mode=455)
    self.assertTrue(os.path.exists('test'))
    self.assertTrue(os.path.exists('test2'))