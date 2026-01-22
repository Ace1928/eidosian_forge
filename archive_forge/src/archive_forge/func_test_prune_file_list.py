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
def test_prune_file_list(self):
    os.mkdir(join(self.tmp_dir, 'somecode', '.svn'))
    self.write_file((self.tmp_dir, 'somecode', '.svn', 'ok.py'), 'xxx')
    os.mkdir(join(self.tmp_dir, 'somecode', '.hg'))
    self.write_file((self.tmp_dir, 'somecode', '.hg', 'ok'), 'xxx')
    os.mkdir(join(self.tmp_dir, 'somecode', '.git'))
    self.write_file((self.tmp_dir, 'somecode', '.git', 'ok'), 'xxx')
    self.write_file((self.tmp_dir, 'somecode', '.nfs0001'), 'xxx')
    dist, cmd = self.get_cmd()
    cmd.formats = ['zip']
    cmd.ensure_finalized()
    cmd.run()
    dist_folder = join(self.tmp_dir, 'dist')
    files = os.listdir(dist_folder)
    self.assertEqual(files, ['fake-1.0.zip'])
    zip_file = zipfile.ZipFile(join(dist_folder, 'fake-1.0.zip'))
    try:
        content = zip_file.namelist()
    finally:
        zip_file.close()
    expected = ['', 'PKG-INFO', 'README', 'setup.py', 'somecode/', 'somecode/__init__.py']
    self.assertEqual(sorted(content), ['fake-1.0/' + x for x in expected])