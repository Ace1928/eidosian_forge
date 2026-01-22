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
def test_minimum_path_selection(self):
    self.assertEqual(set(), osutils.minimum_path_selection([]))
    self.assertEqual({'a'}, osutils.minimum_path_selection(['a']))
    self.assertEqual({'a', 'b'}, osutils.minimum_path_selection(['a', 'b']))
    self.assertEqual({'a/', 'b'}, osutils.minimum_path_selection(['a/', 'b']))
    self.assertEqual({'a/', 'b'}, osutils.minimum_path_selection(['a/c', 'a/', 'b']))
    self.assertEqual({'a-b', 'a', 'a0b'}, osutils.minimum_path_selection(['a-b', 'a/b', 'a0b', 'a']))