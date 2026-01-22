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
def test_resource_string(self):
    text = osutils.resource_string('breezy', 'debug.py')
    self.assertContainsRe(text, 'debug_flags = set()')
    text = osutils.resource_string('breezy.ui', 'text.py')
    self.assertContainsRe(text, 'class TextUIFactory')
    self.assertRaises(errors.BzrError, osutils.resource_string, 'zzzz', 'yyy.xx')
    self.assertRaises(IOError, osutils.resource_string, 'breezy', 'yyy.xx')