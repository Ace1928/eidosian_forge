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
@unittest.skipUnless(ZLIB_SUPPORT, 'Requires zlib')
@unittest.skipUnless(UID_GID_SUPPORT, 'Requires grp and pwd support')
def test_tarfile_root_owner(self):
    tmpdir = self._create_files()
    base_name = os.path.join(self.mkdtemp(), 'archive')
    old_dir = os.getcwd()
    os.chdir(tmpdir)
    group = grp.getgrgid(0)[0]
    owner = pwd.getpwuid(0)[0]
    try:
        archive_name = make_tarball(base_name, 'dist', compress=None, owner=owner, group=group)
    finally:
        os.chdir(old_dir)
    self.assertTrue(os.path.exists(archive_name))
    archive = tarfile.open(archive_name)
    try:
        for member in archive.getmembers():
            self.assertEqual(member.uid, 0)
            self.assertEqual(member.gid, 0)
    finally:
        archive.close()