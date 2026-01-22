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
def test_manifest_comments(self):
    contents = dedent('            # bad.py\n            #bad.py\n            good.py\n            ')
    dist, cmd = self.get_cmd()
    cmd.ensure_finalized()
    self.write_file((self.tmp_dir, cmd.manifest), contents)
    self.write_file((self.tmp_dir, 'good.py'), '# pick me!')
    self.write_file((self.tmp_dir, 'bad.py'), "# don't pick me!")
    self.write_file((self.tmp_dir, '#bad.py'), "# don't pick me!")
    cmd.run()
    self.assertEqual(cmd.filelist.files, ['good.py'])