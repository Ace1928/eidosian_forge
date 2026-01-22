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
def test_make_archive_owner_group(self):
    if UID_GID_SUPPORT:
        group = grp.getgrgid(0)[0]
        owner = pwd.getpwuid(0)[0]
    else:
        group = owner = 'root'
    base_dir = self._create_files()
    root_dir = self.mkdtemp()
    base_name = os.path.join(self.mkdtemp(), 'archive')
    res = make_archive(base_name, 'zip', root_dir, base_dir, owner=owner, group=group)
    self.assertTrue(os.path.exists(res))
    res = make_archive(base_name, 'zip', root_dir, base_dir)
    self.assertTrue(os.path.exists(res))
    res = make_archive(base_name, 'tar', root_dir, base_dir, owner=owner, group=group)
    self.assertTrue(os.path.exists(res))
    res = make_archive(base_name, 'tar', root_dir, base_dir, owner='kjhkjhkjg', group='oihohoh')
    self.assertTrue(os.path.exists(res))