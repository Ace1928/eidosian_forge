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
def test_copy_tree_symlinks(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    self.build_tree(['source/'])
    os.symlink('a/generic/path', 'source/lnk')
    osutils.copy_tree('source', 'target')
    self.assertEqual(['lnk'], os.listdir('target'))
    self.assertEqual('a/generic/path', os.readlink('target/lnk'))