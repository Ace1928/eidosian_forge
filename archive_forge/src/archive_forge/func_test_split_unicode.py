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
def test_split_unicode(self):
    self.assertEqual(['foo\n', 'bar®'], osutils.split_lines('foo\nbar®'))
    self.assertEqual(['foo\n', 'bar®\n'], osutils.split_lines('foo\nbar®\n'))