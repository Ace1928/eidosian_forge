import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def test_walkdirs_os_error(self):
    if sys.platform == 'win32':
        raise tests.TestNotApplicable('readdir IOError not tested on win32')
    self.requireFeature(features.not_running_as_root)
    os.mkdir('test-unreadable')
    os.chmod('test-unreadable', 0)
    self.addCleanup(os.chmod, 'test-unreadable', 448)
    e = self.assertRaises(OSError, list, osutils._walkdirs_utf8('.'))
    self.assertEqual('./test-unreadable', osutils.safe_unicode(e.filename))
    self.assertEqual(errno.EACCES, e.errno)
    self.assertContainsRe(str(e), '\\./test-unreadable')