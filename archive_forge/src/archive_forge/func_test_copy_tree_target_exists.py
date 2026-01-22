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
def test_copy_tree_target_exists(self):
    self.build_tree(['source/', 'source/a', 'source/b/', 'source/b/c', 'target/'])
    osutils.copy_tree('source', 'target')
    self.assertEqual(['a', 'b'], sorted(os.listdir('target')))
    self.assertEqual(['c'], os.listdir('target/b'))