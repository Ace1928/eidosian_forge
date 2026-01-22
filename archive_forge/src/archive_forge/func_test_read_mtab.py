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
def test_read_mtab(self):
    self.build_tree_contents([('mtab', '/dev/mapper/blah--vg-root / ext4 rw,relatime,errors=remount-ro 0 0\n/dev/mapper/blah--vg-home /home vfat rw,relatime 0 0\n# comment\n\niminvalid\n')])
    self.assertEqual(list(osutils.read_mtab('mtab')), [(b'/', 'ext4'), (b'/home', 'vfat')])