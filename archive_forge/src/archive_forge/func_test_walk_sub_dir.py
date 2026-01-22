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
def test_walk_sub_dir(self):
    tree, expected_dirblocks = self._get_ascii_tree()
    self.build_tree(tree)
    result = list(osutils._walkdirs_utf8(b'./1dir', b'1dir'))
    self.assertEqual(expected_dirblocks[1:], self._filter_out(result))