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
def test_rename_current_dir(self):
    os.mkdir('a')
    os.chdir('a')
    try:
        osutils._win32_rename('b', '.')
    except OSError as e:
        self.assertEqual(errno.ENOENT, e.errno)