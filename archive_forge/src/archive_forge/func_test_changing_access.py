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
def test_changing_access(self):
    with open('file', 'w') as f:
        f.write('monkey')
    osutils.make_readonly('file')
    mode = os.lstat('file').st_mode
    self.assertEqual(mode, mode & 261997)
    osutils.make_writable('file')
    mode = os.lstat('file').st_mode
    self.assertEqual(mode, mode | 128)
    if osutils.supports_symlinks(self.test_dir):
        os.symlink('nonexistent', 'dangling')
        osutils.make_readonly('dangling')
        osutils.make_writable('dangling')