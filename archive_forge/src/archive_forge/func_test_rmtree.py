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
def test_rmtree(self):
    os.mkdir('dir')
    with open('dir/file', 'w') as f:
        f.write('spam')
    osutils.make_readonly('dir/file')
    osutils.rmtree('dir')
    self.assertPathDoesNotExist('dir/file')
    self.assertPathDoesNotExist('dir')