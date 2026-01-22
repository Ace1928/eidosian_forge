import unittest
import os
import sys
import tarfile
from os.path import splitdrive
import warnings
from distutils import archive_util
from distutils.archive_util import (check_archive_formats, make_tarball,
from distutils.spawn import find_executable, spawn
from distutils.tests import support
from test.support import patch
from test.support.os_helper import change_cwd
from test.support.warnings_helper import check_warnings
@unittest.skipUnless(ZLIB_SUPPORT, 'Need zlib support to run')
def test_make_archive_gztar(self):
    base_dir = self._create_files()
    base_name = os.path.join(self.mkdtemp(), 'archive')
    res = make_archive(base_name, 'gztar', base_dir, 'dist')
    self.assertTrue(os.path.exists(res))
    self.assertEqual(os.path.basename(res), 'archive.tar.gz')
    self.assertEqual(self._tarinfo(res), self._created_files)