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
def test_make_archive_cwd(self):
    current_dir = os.getcwd()

    def _breaks(*args, **kw):
        raise RuntimeError()
    ARCHIVE_FORMATS['xxx'] = (_breaks, [], 'xxx file')
    try:
        try:
            make_archive('xxx', 'xxx', root_dir=self.mkdtemp())
        except:
            pass
        self.assertEqual(os.getcwd(), current_dir)
    finally:
        del ARCHIVE_FORMATS['xxx']