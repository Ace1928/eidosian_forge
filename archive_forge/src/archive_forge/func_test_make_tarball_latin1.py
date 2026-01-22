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
@unittest.skipUnless(can_fs_encode('årchiv'), 'File system cannot handle this filename')
def test_make_tarball_latin1(self):
    """
        Mirror test_make_tarball, except filename contains latin characters.
        """
    self.test_make_tarball('årchiv')