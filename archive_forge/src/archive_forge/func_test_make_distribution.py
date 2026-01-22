import os
import tarfile
import unittest
import warnings
import zipfile
from os.path import join
from textwrap import dedent
from test.support import captured_stdout
from test.support.warnings_helper import check_warnings
from distutils.command.sdist import sdist, show_formats
from distutils.core import Distribution
from distutils.tests.test_config import BasePyPIRCCommandTestCase
from distutils.errors import DistutilsOptionError
from distutils.spawn import find_executable
from distutils.log import WARN
from distutils.filelist import FileList
from distutils.archive_util import ARCHIVE_FORMATS
from distutils.core import setup
import somecode
@unittest.skipUnless(ZLIB_SUPPORT, 'Need zlib support to run')
@unittest.skipIf(find_executable('tar') is None, 'The tar command is not found')
@unittest.skipIf(find_executable('gzip') is None, 'The gzip command is not found')
def test_make_distribution(self):
    dist, cmd = self.get_cmd()
    cmd.formats = ['gztar', 'tar']
    cmd.ensure_finalized()
    cmd.run()
    dist_folder = join(self.tmp_dir, 'dist')
    result = os.listdir(dist_folder)
    result.sort()
    self.assertEqual(result, ['fake-1.0.tar', 'fake-1.0.tar.gz'])
    os.remove(join(dist_folder, 'fake-1.0.tar'))
    os.remove(join(dist_folder, 'fake-1.0.tar.gz'))
    cmd.formats = ['tar', 'gztar']
    cmd.ensure_finalized()
    cmd.run()
    result = os.listdir(dist_folder)
    result.sort()
    self.assertEqual(result, ['fake-1.0.tar', 'fake-1.0.tar.gz'])