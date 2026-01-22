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
def test_02_rand_chars_100(self):
    result = osutils.rand_chars(100)
    self.assertEqual(len(result), 100)
    self.assertEqual(type(result), str)
    self.assertContainsRe(result, '^[a-z0-9]{100}$')