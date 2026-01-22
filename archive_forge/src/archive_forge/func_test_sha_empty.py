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
def test_sha_empty(self):
    self.build_tree_contents([('foo', b'')])
    expected_sha = osutils.sha_string(b'')
    self.assertEqual(expected_sha, osutils.sha_file_by_name('foo'))